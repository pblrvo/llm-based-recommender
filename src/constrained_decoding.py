"""Trie-based constrained decoding for the grounding tasks (name<->semantic
ID) and any other task whose output is a single item's semantic ID
(sequential, similar_item).

Both the semantic-ID space and the item-name space are closed, enumerable
catalogs (~93k items), not free text. A prefix trie built from every valid
target string, combined with HF `generate()`'s `prefix_allowed_tokens_fn`,
makes the model structurally incapable of emitting a semantic ID or name
that doesn't correspond to a real catalog item -- every generated token is
masked to only the trie's valid next-tokens at that point.

This does not fix whether the model's *ranking* over valid items is
correct -- only that every output is one of them. Useful as a second,
constrained exact-match metric alongside the existing unconstrained one.

Also includes `hierarchical_match`, a prefix-accuracy metric (eugeneyan-
style) for the sid-output tasks: how much of the 4-level RQ-VAE code is
correct before the first mismatch, rather than only all-or-nothing exact
match -- since pure exact-match can't distinguish "got the coarse cluster
right, missed the last digit" from "got nothing right."
"""

import math
import re
from pathlib import Path
from typing import Dict, List, Optional

import polars as pl
import torch

SID_START = "<|sid_start|>"
SID_END = "<|sid_end|>"


class Trie:
    """Token-ID prefix tree. Node = dict[token_id -> node]; END marks a
    valid stopping point at that node (so strings that are strict prefixes
    of other strings in the trie still work)."""

    END = -1

    def __init__(self):
        self.root: dict = {}

    def insert(self, token_ids: List[int]):
        node = self.root
        for tok in token_ids:
            node = node.setdefault(tok, {})
        node[self.END] = {}

    def children_of(self, prefix: List[int]) -> Optional[dict]:
        node = self.root
        for tok in prefix:
            node = node.get(tok)
            if node is None:
                return None
        return node


def load_catalog(project_root: Path, restrict_to_interacted_items: bool = True) -> pl.DataFrame:
    """Item id -> semantic-ID codes, name, and genres, joined the same way
    as build_finetune_dataset.py so the trie's targets match the strings
    the model was actually trained on.

    `restrict_to_interacted_items` defaults to True to match
    build_finetune_dataset.py's default: training data is restricted to the
    ~8.5k items appearing in a user sequence, not the full 93k-item catalog,
    so eval needs the same restriction -- otherwise the trie/validity sets
    would include ~85k items the model was never trained to recognize,
    making "valid" and "constrained exact-match" checks misleadingly
    stricter (or the constrained decoder would offer completions the model
    has no chance of having learned)."""
    sid_df = pl.read_parquet(project_root / "data" / "output" / "semantic_ids.parquet")
    catalog_df = pl.read_parquet(
        project_root / "data" / "clean_game_catalog.parquet", columns=["id", "Name", "Genres"]
    )
    joined = sid_df.join(catalog_df, on="id", how="inner")

    if restrict_to_interacted_items:
        sequences_df = pl.read_parquet(project_root / "data" / "clean_user_sequences.parquet")
        interacted_ids = set()
        for row in sequences_df.iter_rows(named=True):
            interacted_ids.update(row["item_sequence"])
        joined = joined.filter(pl.col("id").is_in(interacted_ids))

    return joined


def semantic_id_to_tokens(semantic_id: List[int]) -> str:
    levels = "".join(f"<|sid_L{level}_{code}|>" for level, code in enumerate(semantic_id))
    return f"{SID_START}{levels}{SID_END}"


def item_description(name: str, genres: Optional[str]) -> str:
    genres = genres.replace(",", ", ") if genres else None
    return f"{name} — {genres}" if genres else name


def build_sid_trie(tokenizer, catalog: pl.DataFrame) -> Trie:
    """All valid `<|sid_start|>...<|sid_end|>` sequences -- used to
    constrain grounding_name2id, sequential, and similar_item outputs."""
    trie = Trie()
    for row in catalog.iter_rows(named=True):
        tokens_str = semantic_id_to_tokens(row["semantic_ids"])
        token_ids = tokenizer(tokens_str, add_special_tokens=False)["input_ids"]
        trie.insert(token_ids)
    return trie


def build_name_trie(tokenizer, catalog: pl.DataFrame) -> Trie:
    """All valid "Name — Genres" descriptions -- used to constrain
    grounding_id2name outputs."""
    trie = Trie()
    for row in catalog.iter_rows(named=True):
        desc = item_description(row["Name"], row["Genres"])
        token_ids = tokenizer(desc, add_special_tokens=False)["input_ids"]
        trie.insert(token_ids)
    return trie


def make_prefix_allowed_tokens_fn(trie: Trie, prompt_len: int, eos_token_id: int):
    """Factory for HF `generate()`'s `prefix_allowed_tokens_fn`. Assumes
    batch size 1 (matches this project's per-example eval loop) since
    prompt_len is fixed for the whole call; batch_id is accepted but
    unused."""

    def fn(batch_id, input_ids):
        generated = input_ids[prompt_len:].tolist()
        node = trie.children_of(generated)
        if node is None:
            # Desynced from the trie (shouldn't happen if this fn drove
            # every step) -- end the sequence rather than return an empty,
            # illegal allow-list.
            return [eos_token_id]
        allowed = [tok for tok in node if tok != Trie.END]
        if Trie.END in node:
            allowed.append(eos_token_id)
        return allowed

    return fn


@torch.no_grad()
def constrained_generate(
    model, tokenizer, prompt: str, trie: Trie, max_new_tokens: int = 32, temperature: Optional[float] = None,
) -> str:
    """Decodes `prompt`, restricted at every step to trie-valid
    continuations. Greedy (deterministic) by default; pass `temperature` to
    sample instead -- e.g. 0.7 for some variety, higher for more. Returns
    the completion with special tokens kept (matching this project's
    existing eval convention), eos_token stripped."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]
    allowed_fn = make_prefix_allowed_tokens_fn(trie, prompt_len, tokenizer.eos_token_id)

    sampling_kwargs = {"do_sample": False}
    if temperature is not None:
        sampling_kwargs = {"do_sample": True, "temperature": temperature}

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        num_beams=1,
        prefix_allowed_tokens_fn=allowed_fn,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        **sampling_kwargs,
    )
    new_tokens = output_ids[0][prompt_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=False).replace(tokenizer.eos_token, "").strip()


@torch.no_grad()
def constrained_beam_search(
    model, tokenizer, prompt: str, trie: Trie, num_beams: int, max_new_tokens: int = 32,
    temperature: Optional[float] = None,
) -> List[str]:
    """Like `constrained_generate`, but returns `num_beams` candidate
    completions via beam search instead of one greedy/sampled completion --
    the ranking equivalent of a classic recommender's top-K scored items,
    since a single greedy decode is a top-1 prediction, not a ranking.
    Ordered best-first (beam score). Used for Recall@K/NDCG@K, computed the
    same way as TIGER (Rajput et al. 2023) and LC-Rec evaluate semantic-ID
    generative recommenders: constrained beam search standing in for the
    ranking step a traditional model gets from a dot-product over all
    items.

    Deterministic beam search by default; pass `temperature` to switch to
    beam-search multinomial sampling (each beam samples from the
    temperature-scaled distribution instead of always expanding the
    highest-probability continuations) -- trades some ranking precision for
    beam diversity, which matters when deterministic beam search's beams
    tend to collapse onto near-identical high-probability variants of the
    same prefix rather than covering genuinely different candidates."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]
    allowed_fn = make_prefix_allowed_tokens_fn(trie, prompt_len, tokenizer.eos_token_id)

    sampling_kwargs = {"do_sample": False}
    if temperature is not None:
        sampling_kwargs = {"do_sample": True, "temperature": temperature}

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        num_return_sequences=num_beams,
        prefix_allowed_tokens_fn=allowed_fn,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        early_stopping=True,
        **sampling_kwargs,
    )
    completions = []
    for seq in output_ids:
        new_tokens = seq[prompt_len:]
        completions.append(tokenizer.decode(new_tokens, skip_special_tokens=False).replace(tokenizer.eos_token, "").strip())
    return completions


_SID_LEVEL_TOKEN = re.compile(r"<\|sid_L(\d)_(\d+)\|>")


def parse_sid_codes(text: str) -> Optional[List[int]]:
    """Extracts the 4 level codes from a '<|sid_start|>...<|sid_end|>'
    string, ordered [L0, L1, L2, L3]. None if `text` doesn't contain
    exactly one code per level 0-3 (malformed/truncated output)."""
    matches = _SID_LEVEL_TOKEN.findall(text)
    if len(matches) != 4:
        return None
    by_level = {int(level): int(code) for level, code in matches}
    if set(by_level) != {0, 1, 2, 3}:
        return None
    return [by_level[level] for level in range(4)]


def hierarchical_match(predicted: str, expected: str) -> Dict[str, bool]:
    """Prefix-accuracy at each RQ-VAE hierarchy level: how much of the code
    sequence is correct before the first mismatch. `l0123` is equivalent to
    plain exact-match; `l0`/`l01`/`l012` show partial credit pure exact-match
    can't -- e.g. "right coarse cluster, wrong disambiguation digit" vs.
    "wrong from the first code"."""
    pred_codes = parse_sid_codes(predicted)
    exp_codes = parse_sid_codes(expected)
    result = {"valid_format": pred_codes is not None, "l0": False, "l01": False, "l012": False, "l0123": False}
    if pred_codes is None or exp_codes is None:
        return result
    result["l0"] = pred_codes[0] == exp_codes[0]
    result["l01"] = pred_codes[:2] == exp_codes[:2]
    result["l012"] = pred_codes[:3] == exp_codes[:3]
    result["l0123"] = pred_codes == exp_codes
    return result


def recall_at_k(candidates: List[str], target: str, k: int) -> float:
    """1.0 if `target` appears among the top-`k` `candidates` (ordered
    best-first, e.g. by beam score), else 0.0. `candidates` may have fewer
    than k entries (e.g. beam search returned fewer valid completions than
    requested) -- treated as-is, no padding needed since membership doesn't
    depend on length."""
    return 1.0 if target in candidates[:k] else 0.0


def ndcg_at_k(candidates: List[str], target: str, k: int) -> float:
    """Binary-relevance NDCG@k: 1/log2(rank+1) if `target` is within the
    top-k of `candidates` (rank is 1-indexed), else 0.0. With exactly one
    relevant item per query, the ideal DCG is always 1 (a single relevant
    item at rank 1), so this reduces to plain DCG@k."""
    top_k = candidates[:k]
    if target not in top_k:
        return 0.0
    rank = top_k.index(target) + 1
    return 1.0 / math.log2(rank + 1)
