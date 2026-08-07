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

from build_finetune_dataset import AlpacaDatasetBuilder

SID_START = "<|sid_start|>"
SID_END = "<|sid_end|>"


class Trie:
    """Token-ID prefix tree for HF generate's prefix_allowed_tokens_fn.

    Each node is a dict[token_id -> node]; END marks a valid stopping point
    at that node (so strings that are strict prefixes of other strings in
    the trie still work).
    """

    END = -1

    def __init__(self):
        """Initialize an empty trie."""
        self.root: dict = {}

    def insert(self, token_ids: List[int]):
        """Add a single token-id sequence as a valid path ending at END."""
        node = self.root
        for tok in token_ids:
            node = node.setdefault(tok, {})
        node[self.END] = {}

    def children_of(self, prefix: List[int]) -> Optional[dict]:
        """Return the trie node reached by walking `prefix`, or None if no such path."""
        node = self.root
        for tok in prefix:
            node = node.get(tok)
            if node is None:
                return None
        return node


def load_catalog(project_root: Path, restrict_to_interacted_items: bool = True) -> pl.DataFrame:
    """Load joined id -> semantic-ID, name, genres, descriptions for the trie.

    Matches build_finetune_dataset.py's catalog-restriction default
    (`restrict_to_interacted_items=True`) so the trie's targets match the
    strings the model was actually trained on.

    Args:
        project_root: Repository root (the directory containing `data/`).
        restrict_to_interacted_items: If True, keep only items appearing in a user sequence.

    Returns:
        Polars DataFrame with columns from semantic_ids + catalog, restricted per the flag.
    """
    sid_df = pl.read_parquet(project_root / "data" / "output" / "semantic_ids.parquet")
    catalog_df = pl.read_parquet(
        project_root / "data" / "clean_game_catalog.parquet",
        columns=["id", "Name", "Genres", "Categories", "About the game"],
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
    """Render a semantic ID as its full sid_start ... sid_end token string."""
    levels = "".join(f"<|sid_L{level}_{code}|>" for level, code in enumerate(semantic_id))
    return f"{SID_START}{levels}{SID_END}"


def item_description(name: str, genres: Optional[str], about_the_game: Optional[str] = None) -> str:
    """Render an item's 'Name — Genres' description, optionally with a short blurb appended.

    Matches build_finetune_dataset.py's grounding_id2name output exactly
    (see AlpacaDatasetBuilder._truncate_blurb) so the trie built from this
    stays in sync with what the model was actually trained to produce.
    `about_the_game` defaults to None (no blurb) for callers that only need
    the short form.

    Args:
        name: Item name.
        genres: Comma-separated genre string (raw catalog format), or None.
        about_the_game: Optional blurb text appended after a period if present.

    Returns:
        The rendered description string.
    """
    genres = genres.replace(",", ", ") if genres else None
    desc = f"{name} — {genres}" if genres else name
    if about_the_game is not None:
        blurb = AlpacaDatasetBuilder._truncate_blurb(about_the_game)
        if blurb:
            desc = f"{desc}. {blurb}"
    return desc


def build_name_lookup(catalog: pl.DataFrame) -> Dict[str, str]:
    """Build an item-description string -> plain item Name lookup, covering
    both the short "Name — Genres" form (asy's real target) and the
    blurb-enriched long form (grounding_id2name's real target) -- see
    build_name_trie's docstring for why both need to resolve.

    Lets an evaluator score "did it predict the correct game" independent
    of whether it also reproduced genres/blurb text exactly -- map both a
    beam-search candidate and the target through this before calling
    recall_at_k/ndcg_at_k.
    """
    lookup = {}
    for row in catalog.iter_rows(named=True):
        short_desc = item_description(row["Name"], row["Genres"])
        long_desc = item_description(row["Name"], row["Genres"], row["About the game"])
        lookup[short_desc] = row["Name"]
        lookup[long_desc] = row["Name"]
    return lookup


def build_sid_criteria_lookup(catalog: pl.DataFrame) -> Dict[str, dict]:
    """Build a semantic-ID-string -> genres/categories lookup.

    Used by criteria_satisfied_at_k to check whether a beam-search candidate
    for an nl_preference query actually satisfies the genres/categories it
    was asked for, since that task has many valid targets per query rather
    than one fixed correct answer.
    """
    lookup = {}
    for row in catalog.iter_rows(named=True):
        sid = semantic_id_to_tokens(row["semantic_ids"])
        lookup[sid] = {
            "genres": {g.strip() for g in row["Genres"].split(",")} if row["Genres"] else set(),
            "categories": {c.strip() for c in row["Categories"].split(",")} if row["Categories"] else set(),
        }
    return lookup


def build_sid_trie(tokenizer, catalog: pl.DataFrame) -> Trie:
    """Build a trie over every valid '<|sid_start|>...<|sid_end|>' sequence.

    Used to constrain grounding_name2id, sequential, and similar_item outputs.
    """
    trie = Trie()
    for row in catalog.iter_rows(named=True):
        tokens_str = semantic_id_to_tokens(row["semantic_ids"])
        token_ids = tokenizer(tokens_str, add_special_tokens=False)["input_ids"]
        trie.insert(token_ids)
    return trie


def build_name_trie(tokenizer, catalog: pl.DataFrame) -> Trie:
    """Build a trie over every valid item description, in BOTH forms that
    actually appear as real training targets: the plain "Name — Genres"
    short form (asy's target) and the blurb-enriched "Name — Genres.
    <blurb>" long form (grounding_id2name's target -- see
    build_finetune_dataset.py's build_grounding_examples/build_
    sequential_and_asy_examples, which draw from the same item_desc for
    asy but only add the blurb for grounding_id2name).

    Used to constrain grounding_id2name and asy outputs. Both forms are
    inserted as valid stopping points per item (Trie.END supports a string
    being a valid prefix of another string also in the trie) -- a trie with
    only the long form would force asy's generation past where it's
    actually trained to stop; a trie with only the short form would do the
    same to grounding_id2name (guaranteeing a mismatch against every
    val-set target either way).
    """
    trie = Trie()
    for row in catalog.iter_rows(named=True):
        short_desc = item_description(row["Name"], row["Genres"])
        long_desc = item_description(row["Name"], row["Genres"], row["About the game"])
        for desc in {short_desc, long_desc}:
            token_ids = tokenizer(desc, add_special_tokens=False)["input_ids"]
            trie.insert(token_ids)
    return trie


def make_prefix_allowed_tokens_fn(trie: Trie, prompt_len: int, eos_token_id: int):
    """Build a `prefix_allowed_tokens_fn` closure for HF generate().

    Assumes batch size 1 (matches this project's per-example eval loop)
    since `prompt_len` is fixed for the whole call; `batch_id` is accepted
    but unused.

    Args:
        trie: Constrained-decoding trie.
        prompt_len: Length of the prompt token sequence; only generated tokens (after this) are constrained.
        eos_token_id: EOS token id used as the fallback when the trie is desynced.

    Returns:
        Closure suitable for HF generate's `prefix_allowed_tokens_fn` arg.
    """

    def fn(batch_id, input_ids):
        """Return token ids allowed at this generation step."""
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
    """Decode `prompt`, restricted at every step to trie-valid continuations.

    Greedy (deterministic) by default; pass `temperature` to sample instead.

    Args:
        model: HF causal LM.
        tokenizer: Matching HF tokenizer.
        prompt: Prompt string.
        trie: Constrained-decoding trie.
        max_new_tokens: Maximum number of generated tokens.
        temperature: Sampling temperature; None means greedy decoding.

    Returns:
        The completion with special tokens kept (matching this project's
        existing eval convention), eos_token stripped.
    """
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
    """Like `constrained_generate`, but return `num_beams` candidates via beam search.

    The ranking equivalent of a classic recommender's top-K scored items,
    since a single greedy decode is a top-1 prediction, not a ranking.
    Ordered best-first (beam score). Used for Recall@K/NDCG@K, computed the
    same way as TIGER (Rajput et al. 2023) and LC-Rec evaluate semantic-ID
    generative recommenders: constrained beam search standing in for the
    ranking step a traditional model gets from a dot-product over all
    items.

    Deterministic beam search by default; pass `temperature` to switch to
    beam-search multinomial sampling -- trades some ranking precision for
    beam diversity.

    Args:
        model: HF causal LM.
        tokenizer: Matching HF tokenizer.
        prompt: Prompt string.
        trie: Constrained-decoding trie.
        num_beams: Number of beams (and candidates returned).
        max_new_tokens: Maximum number of generated tokens per candidate.
        temperature: Sampling temperature; None means deterministic beam search.

    Returns:
        List of `num_beams` completion strings, best-first.
    """
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
    """Extract the 4 level codes from a '<|sid_start|>...<|sid_end|>' string.

    Returns codes ordered [L0, L1, L2, L3], or None if `text` doesn't contain
    exactly one code per level 0-3 (malformed/truncated output).
    """
    matches = _SID_LEVEL_TOKEN.findall(text)
    if len(matches) != 4:
        return None
    by_level = {int(level): int(code) for level, code in matches}
    if set(by_level) != {0, 1, 2, 3}:
        return None
    return [by_level[level] for level in range(4)]


def hierarchical_match(predicted: str, expected: str) -> Dict[str, bool]:
    """Compute prefix-accuracy at each RQ-VAE hierarchy level.

    Returns flags for `l0`, `l01`, `l012`, and `l0123` (full exact-match).
    Partial-prefix flags show partial credit pure exact-match can't -- e.g.
    "right coarse cluster, wrong disambiguation digit" vs. "wrong from the
    first code".
    """
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
    """Return 1.0 if `target` appears among the top-`k` `candidates` (best-first), else 0.0.

    `candidates` may have fewer than k entries -- treated as-is, no padding
    needed since membership doesn't depend on length.
    """
    return 1.0 if target in candidates[:k] else 0.0


def ndcg_at_k(candidates: List[str], target: str, k: int) -> float:
    """Binary-relevance NDCG@k: 1/log2(rank+1) if `target` is within top-k, else 0.0.

    With exactly one relevant item per query, the ideal DCG is always 1
    (a single relevant item at rank 1), so this reduces to plain DCG@k.
    """
    top_k = candidates[:k]
    if target not in top_k:
        return 0.0
    rank = top_k.index(target) + 1
    return 1.0 / math.log2(rank + 1)


def _first_criteria_match_rank(candidates: List[str], criteria: dict, sid_criteria_lookup: Dict[str, dict], k: int) -> Optional[int]:
    """Return the 1-indexed rank of the first top-k candidate satisfying every entry in `criteria`.

    Returns None if no candidate in the top-k satisfies the criteria.
    """
    required_genres = set(criteria.get("genres", []))
    required_categories = set(criteria.get("categories", []))
    for rank, candidate in enumerate(candidates[:k], start=1):
        meta = sid_criteria_lookup.get(candidate)
        if meta is None:
            continue  # not a real catalog item -- shouldn't happen under constrained decoding, but don't crash on it
        if required_genres <= meta["genres"] and required_categories <= meta["categories"]:
            return rank
    return None


def criteria_satisfied_at_k(candidates: List[str], criteria: dict, sid_criteria_lookup: Dict[str, dict], k: int) -> float:
    """Return 1.0 if at least one of the top-k candidates satisfies `criteria`, else 0.0.

    Recall@k analog for nl_preference: doesn't check for one specific stored
    target -- many different items can validly satisfy an open-ended
    preference query.
    """
    return 1.0 if _first_criteria_match_rank(candidates, criteria, sid_criteria_lookup, k) is not None else 0.0


def criteria_ndcg_at_k(candidates: List[str], criteria: dict, sid_criteria_lookup: Dict[str, dict], k: int) -> float:
    """Return NDCG@k analog for nl_preference, using criteria-satisfaction as the relevance signal."""
    rank = _first_criteria_match_rank(candidates, criteria, sid_criteria_lookup, k)
    return 1.0 / math.log2(rank + 1) if rank is not None else 0.0