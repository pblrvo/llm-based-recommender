"""Popularity baseline for the same tasks/metrics as evaluate_ranking_metrics.py:
"always recommend the globally most popular items, ignoring the input
entirely" -- the sanity check every recommender should beat before its
Recall@K/NDCG@K numbers mean anything. Beating random chance (see this
project's earlier ~0.12% Recall@10 estimate at this catalog size) only shows
the model learned *something*; beating this baseline shows it's actually
conditioning on the input rather than defaulting to popular answers -- exactly
the failure mode this project's dataset rebalancing was built to prevent (see
build_finetune_dataset.py's module docstring).

Popularity is computed from raw item occurrence counts in
data/clean_user_sequences.parquet -- NOT from SFT training-example target
frequency. That distinction matters: build_finetune_dataset.py's
train_val_split_by_group sends each item's *entire* example group to either
train or val, never both, so train-set target frequency and val-set targets
are structurally disjoint (verified: zero overlap between train/val targets
for `sequential`) -- a popularity baseline built from SFT train targets would
score exactly 0% by construction, regardless of how good or bad it actually
is, which isn't a real comparison. Raw sequence-occurrence counts don't have
this problem since they're independent of the SFT dataset's split entirely.

No model/GPU involved: reuses the identical recall_at_k/ndcg_at_k functions
evaluate_ranking_metrics.py uses, for a directly comparable number.
"""

import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List

import polars as pl

from constrained_decoding import item_description, load_catalog, ndcg_at_k, recall_at_k, semantic_id_to_tokens
from logger import Logger

logger = Logger.get_logger(__name__)

TASKS = ["grounding_name2id", "sequential", "similar_item", "grounding_id2name"]
SID_OUTPUT_TASKS = {"grounding_name2id", "sequential", "similar_item"}
K_VALUES = [5, 10]


def load_examples_by_task(path: Path) -> Dict[str, List[dict]]:
    examples_by_task = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            examples_by_task.setdefault(ex["task"], []).append(ex)
    return examples_by_task


def most_popular_items(project_root: Path, k: int) -> List[int]:
    """The k item ids with the most total occurrences across every user's
    play sequence in data/clean_user_sequences.parquet -- true global
    popularity, independent of the SFT dataset's train/val split."""
    sequences_df = pl.read_parquet(project_root / "data" / "clean_user_sequences.parquet")
    counts = Counter()
    for row in sequences_df.iter_rows(named=True):
        counts.update(row["item_sequence"])
    return [item_id for item_id, _ in counts.most_common(k)]


def evaluate_task(val_sample: List[dict], popular_candidates: List[str]) -> Dict[int, Dict[str, float]]:
    per_k_recall = {k: [] for k in K_VALUES}
    per_k_ndcg = {k: [] for k in K_VALUES}
    for ex in val_sample:
        for k in K_VALUES:
            per_k_recall[k].append(recall_at_k(popular_candidates, ex["output"], k))
            per_k_ndcg[k].append(ndcg_at_k(popular_candidates, ex["output"], k))
    return {
        k: {
            "recall": sum(per_k_recall[k]) / len(per_k_recall[k]),
            "ndcg": sum(per_k_ndcg[k]) / len(per_k_ndcg[k]),
        }
        for k in K_VALUES
    }


def run(project_root: Path, n: int = 500, seed: int = 0) -> Dict[str, Dict[int, Dict[str, float]]]:
    val_by_task = load_examples_by_task(project_root / "data" / "output" / "sft_val.jsonl")

    catalog = load_catalog(project_root)
    sid_by_id = {row["id"]: semantic_id_to_tokens(row["semantic_ids"]) for row in catalog.iter_rows(named=True)}
    desc_by_id = {row["id"]: item_description(row["Name"], row["Genres"]) for row in catalog.iter_rows(named=True)}

    popular_item_ids = most_popular_items(project_root, k=max(K_VALUES) * 2)  # headroom in case some ids lack catalog metadata
    popular_sids = [sid_by_id[i] for i in popular_item_ids if i in sid_by_id][: max(K_VALUES)]
    popular_descs = [desc_by_id[i] for i in popular_item_ids if i in desc_by_id][: max(K_VALUES)]
    logger.info("Top-%d globally popular items (by raw sequence occurrence): %s", max(K_VALUES), popular_item_ids[: max(K_VALUES)])

    random.seed(seed)
    results = {}
    for task in TASKS:
        val_examples = val_by_task.get(task)
        if not val_examples:
            logger.warning("No val examples found for task %r, skipping", task)
            continue

        popular_candidates = popular_sids if task in SID_OUTPUT_TASKS else popular_descs
        sample = random.sample(val_examples, min(n, len(val_examples)))
        logger.info(
            "Evaluating popularity baseline for %s (%d val examples, %d popular candidates)...",
            task, len(sample), len(popular_candidates),
        )
        results[task] = evaluate_task(sample, popular_candidates)

    return results


def format_results(results: Dict[str, Dict[int, Dict[str, float]]]) -> str:
    lines = []
    for task, per_k in results.items():
        lines.append(task + ":")
        for k, metrics in per_k.items():
            lines.append(f"  Recall@{k}={metrics['recall']:.2%}  NDCG@{k}={metrics['ndcg']:.4f}")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-n", type=int, default=500, help="Val examples sampled per task (default: 500)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    results = run(project_root, n=args.n, seed=args.seed)
    print(format_results(results))
