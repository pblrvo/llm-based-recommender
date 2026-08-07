"""Recall@K / NDCG@K for the fine-tuned model's recommendation-shaped
tasks, computed via constrained beam search rather than a single greedy
decode.

A single greedy generation is a top-1 prediction, not a ranking, so plain
exact-match (used elsewhere in this project's eval tooling) can't be
compared against classic RecSys baselines' Recall@K/NDCG@K. The standard
fix in the generative-retrieval-for-recsys literature -- TIGER (Rajput et
al. 2023, "Recommender Systems with Generative Retrieval") and LC-Rec
(Zheng et al. 2023, arXiv 2311.09049, the paper this project's `asy` task
is drawn from) -- is constrained beam search: generate K candidates
restricted to real catalog items via the same trie used for constrained
decoding elsewhere in this project, then score them exactly like a
traditional recommender's top-K ranked list. Beam search stands in for the
ranking step a dot-product-over-all-items model gets for free.

Covers the tasks with a well-defined "one correct target" ranking question:
  - grounding_name2id / sequential / similar_item / nl_similar_item: rank
    candidate semantic IDs via the sid_trie.
  - grounding_id2name / asy: rank candidate name+genres(+blurb)
    descriptions via the name_trie (which accepts both the short
    "Name — Genres" form asy targets and the blurb-enriched long form
    grounding_id2name targets -- see build_name_trie), then score via
    `name_lookup` -- both the candidates and the target are mapped down to
    just the item's plain Name before recall_at_k/ndcg_at_k, so the metric
    asks "did it predict the correct game" rather than requiring an exact
    match on genres/blurb text too.

nl_preference is a different kind of question -- many items can validly
satisfy an open-ended query like "an action game", so there's no single
stored target to check recall against. It's scored with
criteria_satisfied_at_k/criteria_ndcg_at_k instead (genre/category
consistency of the top-k candidates), not recall_at_k/ndcg_at_k.
"""

import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from constrained_decoding import (
    Trie,
    build_name_lookup,
    build_name_trie,
    build_sid_criteria_lookup,
    build_sid_trie,
    constrained_beam_search,
    criteria_ndcg_at_k,
    criteria_satisfied_at_k,
    load_catalog,
    ndcg_at_k,
    recall_at_k,
)
from logger import Logger

logger = Logger.get_logger(__name__)

BASE_MODEL_NAME = "Qwen/Qwen3-4B"
TASK_TRIES = {
    "grounding_name2id": "sid",
    "sequential": "sid",
    "similar_item": "sid",
    "nl_similar_item": "sid",
    "grounding_id2name": "name",
    "asy": "name",
    "nl_preference": "sid",
}
# Tasks scored via name_lookup (candidate/target collapsed to plain item
# Name before recall_at_k/ndcg_at_k) rather than raw exact-match -- both
# produce name+genres(+blurb) text, not a single-token-family output like
# the sid tasks, so "predicted the correct game" is the meaningful
# question, not exact string equality on the full description.
NAME_ONLY_TASKS = {"grounding_id2name", "asy"}
K_VALUES = [5, 10]
# 32 (evaluate_task's default) is exactly right for the 6-token sid outputs
# but far too small for name_trie targets once grounding_id2name includes
# a blurb -- measured up to ~90+ tokens for name+genres+blurb. See
# evaluate_task's docstring for why too-small a value here silently
# produces unmatchable (truncated-before-any-trie-END) candidates rather
# than just shorter ones.
NAME_TASK_MAX_NEW_TOKENS = 96


def load_model(adapter_path: Path):
    """Load the Qwen3-4B base model in 4-bit and attach the LoRA adapter at `adapter_path`.

    Returns:
        (model, tokenizer) pair, ready for `eval()`.
    """
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True, llm_int8_skip_modules=["lm_head"],
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME, dtype=torch.bfloat16, quantization_config=quantization_config,
    )
    base_model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    logger.info("Model + adapter loaded from %s", adapter_path)
    return model, tokenizer


def load_val_examples_by_task(val_path: Path) -> Dict[str, List[dict]]:
    """Read a JSONL file and group examples by their 'task' field."""
    examples_by_task = {}
    with open(val_path, encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            examples_by_task.setdefault(ex["task"], []).append(ex)
    return examples_by_task


def evaluate_task(
    model, tokenizer, trie: Trie, examples: List[dict], num_beams: int, temperature: Optional[float] = None,
    result_lookup: Optional[Dict[str, str]] = None, max_new_tokens: int = 32,
) -> Dict[int, Dict[str, float]]:
    """Run constrained beam search over every example and return mean Recall@k/NDCG@k per K.

    `result_lookup`, if given, maps each candidate and the target through it
    before scoring -- e.g. build_name_lookup, which collapses a full
    grounding_id2name description down to just the item's Name, so the
    metric checks "predicted the correct game" rather than requiring an
    exact match on genres/blurb text too. Entries missing from the lookup
    are left as-is (defensive; every trie-constrained candidate and every
    real target should already be present).

    `max_new_tokens` (default 32) is plenty for the sid-output tasks -- a
    semantic ID is always exactly 6 tokens -- but far too small for
    grounding_id2name/asy's name+genres(+blurb) targets, which can run to
    ~90+ tokens once the blurb is included. Too small a value here doesn't
    just truncate the text -- it can cut generation off *before* the model
    reaches any trie-valid stopping point, producing a candidate that
    never matches any real catalog entry regardless of whether the model
    "knew" the right answer. Callers evaluating a name_trie task should
    pass a larger value (see run()).
    """
    per_k_recall = {k: [] for k in K_VALUES}
    per_k_ndcg = {k: [] for k in K_VALUES}

    for i, ex in enumerate(examples):
        messages = [{"role": "user", "content": f"{ex['instruction']}\n{ex['input']}"}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        candidates = constrained_beam_search(
            model, tokenizer, prompt, trie, num_beams=num_beams, temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
        target = ex["output"]
        if result_lookup is not None:
            candidates = [result_lookup.get(c, c) for c in candidates]
            target = result_lookup.get(target, target)

        for k in K_VALUES:
            per_k_recall[k].append(recall_at_k(candidates, target, k))
            per_k_ndcg[k].append(ndcg_at_k(candidates, target, k))

        if (i + 1) % 5 == 0:
            logger.info("  ...%d/%d examples", i + 1, len(examples))

    return {
        k: {
            "recall": sum(per_k_recall[k]) / len(per_k_recall[k]),
            "ndcg": sum(per_k_ndcg[k]) / len(per_k_ndcg[k]),
        }
        for k in K_VALUES
    }


def evaluate_nl_preference_task(
    model, tokenizer, trie: Trie, examples: List[dict], num_beams: int,
    sid_criteria_lookup: Dict[str, dict], temperature: Optional[float] = None,
) -> Dict[int, Dict[str, float]]:
    """Like `evaluate_task`, but score genre/category consistency instead of exact-match recall."""
    per_k_recall = {k: [] for k in K_VALUES}
    per_k_ndcg = {k: [] for k in K_VALUES}

    for i, ex in enumerate(examples):
        messages = [{"role": "user", "content": f"{ex['instruction']}\n{ex['input']}"}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        candidates = constrained_beam_search(
            model, tokenizer, prompt, trie, num_beams=num_beams, temperature=temperature,
        )

        for k in K_VALUES:
            per_k_recall[k].append(criteria_satisfied_at_k(candidates, ex["criteria"], sid_criteria_lookup, k))
            per_k_ndcg[k].append(criteria_ndcg_at_k(candidates, ex["criteria"], sid_criteria_lookup, k))

        if (i + 1) % 5 == 0:
            logger.info("  ...%d/%d examples", i + 1, len(examples))

    return {
        k: {
            "recall": sum(per_k_recall[k]) / len(per_k_recall[k]),
            "ndcg": sum(per_k_ndcg[k]) / len(per_k_ndcg[k]),
        }
        for k in K_VALUES
    }


def run(
    adapter_path: Path, project_root: Path, n: int = 500, seed: int = 0, temperature: Optional[float] = None,
    source: str = "val",
) -> Dict[str, Dict[int, Dict[str, float]]]:
    """Run Recall@K/NDCG@K evaluation for every task in TASK_TRIES.

    Args:
        adapter_path: LoRA adapter directory.
        project_root: Repository root containing `data/`.
        n: Examples sampled per task.
        seed: RNG seed used to sample examples.
        temperature: Beam-search sampling temperature; None means deterministic.
        source: 'val' (default, held-out) or 'train' (diagnostic only).

    Returns:
        Nested dict {task: {k: {recall, ndcg}}}.
    """
    data_path = project_root / "data" / "output" / ("sft_train.jsonl" if source == "train" else "sft_val.jsonl")
    model, tokenizer = load_model(adapter_path)

    catalog = load_catalog(project_root)
    sid_trie = build_sid_trie(tokenizer, catalog)
    name_trie = build_name_trie(tokenizer, catalog)
    tries = {"sid": sid_trie, "name": name_trie}
    sid_criteria_lookup = build_sid_criteria_lookup(catalog)
    name_lookup = build_name_lookup(catalog)

    examples_by_task = load_val_examples_by_task(data_path)
    num_beams = max(K_VALUES)

    random.seed(seed)
    results = {}
    for task, trie_kind in TASK_TRIES.items():
        examples = examples_by_task.get(task)
        if not examples:
            logger.warning("No validation examples found for task %r, skipping", task)
            continue
        sample = random.sample(examples, min(n, len(examples)))
        logger.info(
            "Evaluating %s (%d %s examples, num_beams=%d, temperature=%s)...",
            task, len(sample), source, num_beams, temperature,
        )
        if task == "nl_preference":
            results[task] = evaluate_nl_preference_task(
                model, tokenizer, tries[trie_kind], sample, num_beams, sid_criteria_lookup, temperature=temperature,
            )
        elif task in NAME_ONLY_TASKS:
            results[task] = evaluate_task(
                model, tokenizer, tries[trie_kind], sample, num_beams, temperature=temperature,
                result_lookup=name_lookup, max_new_tokens=NAME_TASK_MAX_NEW_TOKENS,
            )
        else:
            results[task] = evaluate_task(model, tokenizer, tries[trie_kind], sample, num_beams, temperature=temperature)

    return results


def format_results(results: Dict[str, Dict[int, Dict[str, float]]]) -> str:
    """Render a results dict as a human-readable multi-line string."""
    lines = []
    for task, per_k in results.items():
        lines.append(task + ":")
        for k, metrics in per_k.items():
            lines.append(f"  Recall@{k}={metrics['recall']:.2%}  NDCG@{k}={metrics['ndcg']:.4f}")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-n", type=int, default=500, help="Examples sampled per task (default: 500)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--temperature", type=float, default=None,
        help="Beam-search multinomial sampling temperature. Omit for deterministic beam search.",
    )
    parser.add_argument(
        "--source", choices=["val", "train"], default="val",
        help="'train' evaluates against seen training examples -- a diagnostic, not a real generalization metric (see run()'s docstring).",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    adapter_path = project_root / "models" / "qwen3-4b-qlora"
    results = run(adapter_path, project_root, n=args.n, seed=args.seed, temperature=args.temperature, source=args.source)
    print(format_results(results))