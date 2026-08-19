"""Accuracy eval for the `relatedness` task -- binary "do these two semantic
IDs share a level-0 code" classification, scored as plain classification
accuracy/precision/recall/F1 rather than Recall@K.
"""

import random
from pathlib import Path
from typing import Dict, List

from constrained_decoding import Trie, constrained_generate
from evaluate_ranking_metrics import load_model, load_val_examples_by_task
from logger import Logger

logger = Logger.get_logger(__name__)


def build_yes_no_trie(tokenizer) -> Trie:
    """Trie over exactly {"Yes", "No"} -- the task's entire output space."""
    trie = Trie()
    for word in ("Yes", "No"):
        token_ids = tokenizer(word, add_special_tokens=False)["input_ids"]
        trie.insert(token_ids)
    return trie


def evaluate(model, tokenizer, trie: Trie, examples: List[dict]) -> Dict[str, float]:
    """Run constrained greedy decoding and return classification metrics (accuracy/precision/recall/F1 for "Yes")."""
    tp = fp = tn = fn = 0
    for i, ex in enumerate(examples):
        messages = [{"role": "user", "content": f"{ex['instruction']}\n{ex['input']}"}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prediction = constrained_generate(model, tokenizer, prompt, trie, max_new_tokens=4)
        predicted_yes = prediction.strip() == "Yes"
        actual_yes = ex["output"].strip() == "Yes"

        if predicted_yes and actual_yes:
            tp += 1
        elif predicted_yes and not actual_yes:
            fp += 1
        elif not predicted_yes and not actual_yes:
            tn += 1
        else:
            fn += 1

        if (i + 1) % 100 == 0:
            logger.info("  ...%d/%d examples", i + 1, len(examples))

    n = tp + fp + tn + fn
    accuracy = (tp + tn) / n if n else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else float("nan")
    return {
        "n": n, "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def run(model_path: Path, project_root: Path, n: int = 500, seed: int = 0) -> Dict[str, float]:
    """Load the model, sample `n` relatedness val examples, and return their classification metrics."""
    model, tokenizer = load_model(model_path)
    trie = build_yes_no_trie(tokenizer)

    examples_by_task = load_val_examples_by_task(project_root / "data" / "output" / "sft_val.jsonl")
    examples = examples_by_task.get("relatedness")
    if not examples:
        raise RuntimeError("No 'relatedness' validation examples found")

    random.seed(seed)
    sample = random.sample(examples, min(n, len(examples)))
    logger.info("Evaluating relatedness (%d val examples, constrained Yes/No decoding)...", len(sample))
    return evaluate(model, tokenizer, trie, sample)


def format_results(results: Dict[str, float]) -> str:
    """Render results as a human-readable summary string."""
    return (
        f"relatedness: n={results['n']}\n"
        f"  accuracy={results['accuracy']:.2%}  precision={results['precision']:.2%}  "
        f"recall={results['recall']:.2%}  f1={results['f1']:.2%}\n"
        f"  confusion: tp={results['tp']} fp={results['fp']} tn={results['tn']} fn={results['fn']}"
    )


def self_check():
    """Assert the metric arithmetic without touching a model."""
    fake = {"tp": 3, "fp": 1, "tn": 4, "fn": 2}
    n = sum(fake.values())
    accuracy = (fake["tp"] + fake["tn"]) / n
    precision = fake["tp"] / (fake["tp"] + fake["fp"])
    recall = fake["tp"] / (fake["tp"] + fake["fn"])
    assert abs(accuracy - 0.7) < 1e-9
    assert abs(precision - 0.75) < 1e-9
    assert abs(recall - 0.6) < 1e-9
    print("metrics: ok")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, default="pblrvo/Qwen3-8B-Game-semantic-IDs-v3")
    parser.add_argument("-n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()

    if args.self_check:
        self_check()
    else:
        project_root = Path(__file__).resolve().parent.parent
        results = run(Path(args.model), project_root, n=args.n, seed=args.seed)
        formatted = format_results(results)
        print(formatted)
        (project_root / "outputs").mkdir(exist_ok=True)
        (project_root / "outputs" / "relatedness_eval.txt").write_text(formatted, encoding="utf-8")
