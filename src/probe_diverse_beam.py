"""Small-scale probe for a diverse-beam-search fix to the name_trie collapse
(see constrained_beam_search's docstring). Tests several (num_beam_groups,
diversity_penalty) configs at small n before trusting any at full eval scale.

Run on a GPU pod: `python probe_diverse_beam.py --model /path/to/model`
"""

import random
from pathlib import Path

from constrained_decoding import build_name_lookup, build_name_trie, constrained_beam_search, load_catalog
from evaluate_ranking_metrics import NAME_TASK_MAX_NEW_TOKENS, load_model, load_val_examples_by_task
from logger import Logger

logger = Logger.get_logger(__name__)

# (num_beam_groups, diversity_penalty) configs to try. (1, 0.0) is the plain-beam baseline.
CONFIGS = [
    (1, 0.0),
    (2, 0.2),
    (2, 0.4),
    (5, 0.2),
    (5, 0.4),
]
N_PROBE = 8


def collapse_signature(candidates: list[str], name_lookup: dict) -> tuple[int, bool]:
    """Return (distinct item names among the 10 candidates, all-identical flag)."""
    names = [name_lookup.get(c, c) for c in candidates]
    return len(set(names)), len(set(names)) <= 1


def run(model_path: Path, project_root: Path, n: int = N_PROBE, seed: int = 0):
    """Run every config in CONFIGS against a sample of `asy` val examples and log the collapse stats for each."""
    model, tokenizer = load_model(model_path)
    catalog = load_catalog(project_root)
    name_trie = build_name_trie(tokenizer, catalog)
    name_lookup = build_name_lookup(catalog)

    examples_by_task = load_val_examples_by_task(project_root / "data" / "output" / "sft_val.jsonl")
    random.seed(seed)
    sample = random.sample(examples_by_task["asy"], n)

    for groups, penalty in CONFIGS:
        logger.info("=== num_beam_groups=%d diversity_penalty=%.1f ===", groups, penalty)
        distinct_counts, collapsed_count, degenerate_count = [], 0, 0
        for ex in sample:
            messages = [{"role": "user", "content": f"{ex['instruction']}\n{ex['input']}"}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            candidates = constrained_beam_search(
                model, tokenizer, prompt, name_trie, num_beams=10,
                max_new_tokens=NAME_TASK_MAX_NEW_TOKENS,
                num_beam_groups=groups, diversity_penalty=penalty,
            )
            distinct, collapsed = collapse_signature(candidates, name_lookup)
            distinct_counts.append(distinct)
            collapsed_count += collapsed
            degenerate_count += sum(1 for c in candidates if c not in name_lookup)
            target_name = name_lookup.get(ex["output"], ex["output"])
            hit5 = target_name in [name_lookup.get(c, c) for c in candidates[:5]]
            hit10 = target_name in [name_lookup.get(c, c) for c in candidates[:10]]
            logger.info(
                "  distinct=%d/10  collapsed=%s  target_in_top5=%s  target_in_top10=%s  sample_cand=%r",
                distinct, collapsed, hit5, hit10, candidates[0][:80],
            )
        mean_distinct = sum(distinct_counts) / len(distinct_counts)
        logger.info(
            "  --> mean distinct=%.1f/10, collapsed on %d/%d, degenerate candidates=%d/%d",
            mean_distinct, collapsed_count, n, degenerate_count, n * 10,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, default="pblrvo/Qwen3-8B-Game-semantic-IDs-v3")
    parser.add_argument("-n", type=int, default=N_PROBE)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    run(Path(args.model), project_root, n=args.n, seed=args.seed)
