"""Multi-target scoring for sequential/asy: "did the model recommend ANY
game this user actually played?" instead of "did it name the single
next-lower-playtime game?". A separate script from evaluate_ranking_metrics.py
so its single-target numbers stay untouched; this one reports both metrics
side by side over the same beam-search candidates.

The relevant set (every item a user played, minus the ones shown in the
prompt) is rebuilt from clean_user_sequences.parquet by replaying
build_finetune_dataset.py's own windowing, keyed on the history string.
"""

import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Set

import polars as pl

from constrained_decoding import (
    Trie,
    build_name_lookup,
    build_name_trie,
    build_sid_trie,
    constrained_beam_search,
    load_catalog,
    ndcg_at_k,
    recall_at_k,
    semantic_id_to_tokens,
)
from evaluate_ranking_metrics import (
    K_VALUES,
    NAME_TASK_MAX_NEW_TOKENS,
    load_model,
    load_val_examples_by_task,
)
from logger import Logger

logger = Logger.get_logger(__name__)

TASKS = ("sequential", "asy")
HISTORY_WINDOW = 10  # must match build_finetune_dataset.py's windowing default


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------

def hit_at_k(candidates: List[str], relevant: FrozenSet[str], k: int) -> float:
    """Return 1.0 if any of the top-k candidates is in `relevant`, else 0.0 (the multi-target analog of recall_at_k)."""
    return 1.0 if any(c in relevant for c in candidates[:k]) else 0.0


def random_hit_at_k(n_relevant: int, n_catalog: int, k: int) -> float:
    """Expected hit@k when k distinct items are drawn uniformly at random from the catalog (hypergeometric)."""
    miss = 1.0
    for i in range(k):
        remaining = n_catalog - i
        if remaining <= 0:
            return 1.0
        miss *= max(0.0, (n_catalog - n_relevant - i)) / remaining
    return 1.0 - miss


def multi_ndcg_at_k(candidates: List[str], relevant: FrozenSet[str], k: int) -> float:
    """Binary-relevance NDCG@k with more than one relevant item (normalizer computed, not assumed to be 1.0)."""
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, c in enumerate(candidates[:k], start=1)
        if c in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


# ----------------------------------------------------------------------
# Relevant-set reconstruction
# ----------------------------------------------------------------------

def build_played_lookup(project_root: Path, catalog: pl.DataFrame) -> Dict[str, FrozenSet[int]]:
    """Map each history string in the val set to the user's other played item IDs.

    A history window shared by two different users maps to the union of both
    users' played sets (rare, but handled by taking the union rather than
    picking one).
    """
    known_ids = set(catalog["id"].to_list())
    sequences_df = pl.read_parquet(project_root / "data" / "clean_user_sequences.parquet")

    lookup: Dict[str, Set[int]] = defaultdict(set)
    sid_tokens = {row["id"]: semantic_id_to_tokens(row["semantic_ids"]) for row in catalog.iter_rows(named=True)}

    for row in sequences_df.iter_rows(named=True):
        if row.get("is_long_tail_user"):
            continue
        sequence = [
            item_id
            for item_id, playtime in zip(row["item_sequence"], row["playtime_sequence"])
            if playtime > 0 and item_id in known_ids
        ]
        if len(sequence) < 2:
            continue

        played = set(sequence)
        for pos in range(1, len(sequence)):
            history = sequence[max(0, pos - HISTORY_WINDOW):pos]
            key = " ".join(sid_tokens[i] for i in history)
            lookup[key] |= played - set(history)

    return {k: frozenset(v) for k, v in lookup.items()}


def relevant_strings(
    relevant_ids: FrozenSet[int], task: str, sid_tokens: Dict[int, str], names: Dict[int, str],
) -> FrozenSet[str]:
    """Render a relevant item-ID set in whatever string space `task` is scored in (sid tokens or plain Name)."""
    render = sid_tokens if task == "sequential" else names
    return frozenset(render[i] for i in relevant_ids if i in render)


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------

def evaluate_task_both_metrics(
    model, tokenizer, trie: Trie, examples: List[dict], task: str,
    lookup: Dict[str, FrozenSet[int]], sid_tokens: Dict[int, str], names: Dict[int, str],
    num_beams: int, max_new_tokens: int, result_lookup: Optional[Dict[str, str]] = None,
) -> dict:
    """Beam-search once per example, then score it under both the single- and multi-target metric.

    Examples whose history string isn't in `lookup` are still scored under the
    single-target metric but skipped for the multi-target one, and counted in
    the returned `coverage`.
    """
    single = {k: {"recall": [], "ndcg": []} for k in K_VALUES}
    multi = {k: {"hit": [], "ndcg": []} for k in K_VALUES}
    baseline = {k: [] for k in K_VALUES}
    matched, relevant_sizes = 0, []
    n_catalog = len(names)

    for i, ex in enumerate(examples):
        messages = [{"role": "user", "content": f"{ex['instruction']}\n{ex['input']}"}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        candidates = constrained_beam_search(
            model, tokenizer, prompt, trie, num_beams=num_beams, max_new_tokens=max_new_tokens,
        )
        target = ex["output"]
        if result_lookup is not None:
            candidates = [result_lookup.get(c, c) for c in candidates]
            target = result_lookup.get(target, target)

        for k in K_VALUES:
            single[k]["recall"].append(recall_at_k(candidates, target, k))
            single[k]["ndcg"].append(ndcg_at_k(candidates, target, k))

        relevant_ids = lookup.get(ex["input"])
        if relevant_ids is not None:
            matched += 1
            relevant = relevant_strings(relevant_ids, task, sid_tokens, names)
            relevant = relevant | {target}  # the stored target is relevant by definition
            relevant_sizes.append(len(relevant))
            for k in K_VALUES:
                multi[k]["hit"].append(hit_at_k(candidates, relevant, k))
                multi[k]["ndcg"].append(multi_ndcg_at_k(candidates, relevant, k))
                baseline[k].append(random_hit_at_k(len(relevant), n_catalog, k))

        if (i + 1) % 25 == 0:
            logger.info("  ...%d/%d examples", i + 1, len(examples))

    def mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    # Relevant-set size is heavily skewed; `small` re-scores only the below-median-library half.
    median_size = sorted(relevant_sizes)[len(relevant_sizes) // 2] if relevant_sizes else 0
    small = [i for i, s in enumerate(relevant_sizes) if s <= median_size]

    return {
        "n": len(examples),
        "coverage": matched / len(examples) if examples else 0.0,
        "mean_relevant": mean(relevant_sizes),
        "median_relevant": median_size,
        "n_small": len(small),
        "single": {k: {m: mean(v) for m, v in per_m.items()} for k, per_m in single.items()},
        "multi": {k: {m: mean(v) for m, v in per_m.items()} for k, per_m in multi.items()},
        "random": {k: mean(v) for k, v in baseline.items()},
        "small": {
            k: {
                "hit": mean([multi[k]["hit"][i] for i in small]),
                "random": mean([baseline[k][i] for i in small]),
            }
            for k in K_VALUES
        },
    }


def run(model_path: Path, project_root: Path, n: int = 500, seed: int = 0) -> dict:
    """Score sequential and asy under both metrics and return a nested results dict."""
    model, tokenizer = load_model(model_path)

    catalog = load_catalog(project_root)
    tries = {"sequential": build_sid_trie(tokenizer, catalog), "asy": build_name_trie(tokenizer, catalog)}
    name_lookup = build_name_lookup(catalog)
    sid_tokens = {row["id"]: semantic_id_to_tokens(row["semantic_ids"]) for row in catalog.iter_rows(named=True)}
    names = {row["id"]: row["Name"] for row in catalog.iter_rows(named=True)}

    lookup = build_played_lookup(project_root, catalog)
    logger.info("Built played-item lookup over %d distinct history windows", len(lookup))

    examples_by_task = load_val_examples_by_task(project_root / "data" / "output" / "sft_val.jsonl")

    random.seed(seed)
    results = {}
    for task in TASKS:
        examples = examples_by_task.get(task)
        if not examples:
            logger.warning("No validation examples for task %r, skipping", task)
            continue
        sample = random.sample(examples, min(n, len(examples)))
        logger.info("Evaluating %s (%d examples, num_beams=%d)...", task, len(sample), max(K_VALUES))
        results[task] = evaluate_task_both_metrics(
            model, tokenizer, tries[task], sample, task, lookup, sid_tokens, names,
            num_beams=max(K_VALUES),
            max_new_tokens=NAME_TASK_MAX_NEW_TOKENS if task == "asy" else 32,
            result_lookup=name_lookup if task == "asy" else None,
        )
    return results


def format_results(results: dict) -> str:
    """Render results as a side-by-side single- vs multi-target table."""
    lines = []
    for task, r in results.items():
        lines.append(
            f"{task}: n={r['n']}, coverage={r['coverage']:.1%}, "
            f"relevant items mean={r['mean_relevant']:.1f} median={r['median_relevant']}"
        )
        for k in K_VALUES:
            hit, rand = r["multi"][k]["hit"], r["random"][k]
            lift = hit / rand if rand else float("nan")
            lines.append(
                f"  @{k}:  exact-target Recall={r['single'][k]['recall']:.2%} NDCG={r['single'][k]['ndcg']:.4f}"
                f"   |   any-played Hit={hit:.2%} NDCG={r['multi'][k]['ndcg']:.4f}"
                f"   (random={rand:.2%}, lift={lift:.2f}x)"
            )
        for k in K_VALUES:
            s = r["small"][k]
            lift = s["hit"] / s["random"] if s["random"] else float("nan")
            lines.append(
                f"  @{k} small-library half (n={r['n_small']}, <={r['median_relevant']} relevant): "
                f"Hit={s['hit']:.2%} (random={s['random']:.2%}, lift={lift:.2f}x)"
            )
    return "\n".join(lines)


# ----------------------------------------------------------------------
# Self-check (no GPU, no model -- run this before spending GPU time)
# ----------------------------------------------------------------------

def self_check(project_root: Path):
    """Assert the metrics behave, then verify the lookup actually covers the real val set."""
    assert hit_at_k(["a", "b", "c"], frozenset({"c"}), 3) == 1.0
    assert hit_at_k(["a", "b", "c"], frozenset({"c"}), 2) == 0.0
    assert hit_at_k(["a"], frozenset({"x", "y"}), 5) == 0.0
    assert hit_at_k(["a", "b"], frozenset({"b"}), 2) == recall_at_k(["a", "b"], "b", 2)
    assert abs(multi_ndcg_at_k(["a", "b"], frozenset({"b"}), 2) - ndcg_at_k(["a", "b"], "b", 2)) < 1e-12
    assert abs(multi_ndcg_at_k(["a", "b", "c"], frozenset({"a", "b"}), 3) - 1.0) < 1e-12
    assert multi_ndcg_at_k(["a", "b"], frozenset(), 2) == 0.0
    assert multi_ndcg_at_k(["a", "b"], frozenset({"a"}), 2) > multi_ndcg_at_k(["b", "a"], frozenset({"a"}), 2)
    assert random_hit_at_k(0, 100, 10) == 0.0
    assert random_hit_at_k(100, 100, 1) == 1.0
    assert abs(random_hit_at_k(1, 100, 1) - 0.01) < 1e-12
    assert random_hit_at_k(5, 100, 10) > random_hit_at_k(5, 100, 5)
    print("metrics: ok")

    catalog = load_catalog(project_root)
    lookup = build_played_lookup(project_root, catalog)
    examples_by_task = load_val_examples_by_task(project_root / "data" / "output" / "sft_val.jsonl")

    for task in TASKS:
        examples = examples_by_task.get(task, [])
        matched = [lookup[ex["input"]] for ex in examples if ex["input"] in lookup]
        coverage = len(matched) / len(examples) if examples else 0.0
        assert coverage > 0.95, f"{task} lookup coverage {coverage:.1%} -- windowing has drifted from the dataset builder"
        sizes = sorted(len(m) for m in matched)
        n_catalog = catalog.height
        rand = {k: sum(random_hit_at_k(s, n_catalog, k) for s in sizes) / len(sizes) for k in K_VALUES}
        print(
            f"{task}: {len(examples)} val examples, coverage {coverage:.1%}, "
            f"relevant items mean={sum(sizes) / len(sizes):.1f} median={sizes[len(sizes) // 2]} "
            f"max={sizes[-1]} ({sum(sizes) / len(sizes) / n_catalog:.2%} of {n_catalog}-item catalog)"
        )
        print("  chance-level any-played Hit: " + "  ".join(f"@{k}={rand[k]:.2%}" for k in K_VALUES))
    print("lookup: ok")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, default="pblrvo/Qwen3-8B-Game-semantic-IDs-v3")
    parser.add_argument("-n", type=int, default=500, help="Examples sampled per task (default: 500)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--self-check", action="store_true", help="Validate metrics + lookup coverage without a GPU, then exit.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    if args.self_check:
        self_check(project_root)
    else:
        results = run(Path(args.model), project_root, n=args.n, seed=args.seed)
        print(format_results(results))
        (project_root / "outputs").mkdir(exist_ok=True)
        (project_root / "outputs" / "multi_target_eval.txt").write_text(format_results(results), encoding="utf-8")
