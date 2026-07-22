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

Covers the four tasks with a well-defined "one correct target" ranking
question:
  - grounding_name2id / sequential / similar_item: rank candidate semantic
    IDs via the sid_trie.
  - grounding_id2name: rank candidate name+genres descriptions via the
    name_trie.
`asy` is skipped -- its target is the same (history -> name) question as
grounding_id2name/sequential combined, not a distinct ranking question.
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
    build_name_trie,
    build_sid_trie,
    constrained_beam_search,
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
    "grounding_id2name": "name",
}
K_VALUES = [5, 10]


def load_model(adapter_path: Path):
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
    examples_by_task = {}
    with open(val_path, encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            examples_by_task.setdefault(ex["task"], []).append(ex)
    return examples_by_task


def evaluate_task(
    model, tokenizer, trie: Trie, examples: List[dict], num_beams: int, temperature: Optional[float] = None,
) -> Dict[int, Dict[str, float]]:
    """Runs constrained beam search over every example and returns
    {k: {"recall": mean_recall_at_k, "ndcg": mean_ndcg_at_k}} for each K in
    K_VALUES."""
    per_k_recall = {k: [] for k in K_VALUES}
    per_k_ndcg = {k: [] for k in K_VALUES}

    for i, ex in enumerate(examples):
        messages = [{"role": "user", "content": f"{ex['instruction']}\n{ex['input']}"}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        candidates = constrained_beam_search(
            model, tokenizer, prompt, trie, num_beams=num_beams, temperature=temperature,
        )

        for k in K_VALUES:
            per_k_recall[k].append(recall_at_k(candidates, ex["output"], k))
            per_k_ndcg[k].append(ndcg_at_k(candidates, ex["output"], k))

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
) -> Dict[str, Dict[int, Dict[str, float]]]:
    val_path = project_root / "data" / "output" / "sft_val.jsonl"
    model, tokenizer = load_model(adapter_path)

    catalog = load_catalog(project_root)
    sid_trie = build_sid_trie(tokenizer, catalog)
    name_trie = build_name_trie(tokenizer, catalog)
    tries = {"sid": sid_trie, "name": name_trie}

    examples_by_task = load_val_examples_by_task(val_path)
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
            "Evaluating %s (%d examples, num_beams=%d, temperature=%s)...",
            task, len(sample), num_beams, temperature,
        )
        results[task] = evaluate_task(model, tokenizer, tries[trie_kind], sample, num_beams, temperature=temperature)

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
    parser.add_argument("-n", type=int, default=500, help="Examples sampled per task (default: 500)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--temperature", type=float, default=None,
        help="Beam-search multinomial sampling temperature. Omit for deterministic beam search.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    adapter_path = project_root / "models" / "qwen3-4b-qlora"
    results = run(adapter_path, project_root, n=args.n, seed=args.seed, temperature=args.temperature)
    print(format_results(results))
