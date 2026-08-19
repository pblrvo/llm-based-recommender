"""Orchestrates the full Stage 1 (embedding warmup) + Stage 2 (full-parameter
fine-tune) + eval + Hugging Face Hub push run for the Qwen3-8B model.

`warmup_embeddings` is imported locally inside run_stage1(), not at module
level: it imports unsloth, which monkey-patches transformers/trl process-wide
at import time and breaks full_finetune_8b.py's plain SFTTrainer if loaded
first. Keeping the import local means run_stage2()/run_eval() alone never
pulls unsloth in.
"""

import gc
from pathlib import Path

import torch

import evaluate_ranking_metrics
from full_finetune_8b import FullFineTuneConfig, FullFineTuneTrainer
from logger import Logger

logger = Logger.get_logger(__name__)

STAGE1_OUTPUT_DIR = Path("outputs/qwen3-8b-embed-warmup")
STAGE2_OUTPUT_DIR = Path("outputs/qwen3-8b-full-finetune")
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Set to None to skip the Hugging Face push.
HF_REPO_ID = "pblrvo/Qwen3-8B-Game-semantic-IDs-v3"


def run_stage1():
    """Run Stage 1 (embedding warmup) with the Qwen3-8B / non-quantized configuration and return its output dir."""
    from warmup_embeddings import EmbeddingWarmupConfig, EmbeddingWarmupTrainer  # local: see module docstring

    logger.info("=== Stage 1: embedding warmup (Qwen3-8B, non-quantized) ===")
    config = EmbeddingWarmupConfig(
        base_model="Qwen/Qwen3-8B",
        load_in_4bit=False,
        output_dir=STAGE1_OUTPUT_DIR,
    )
    EmbeddingWarmupTrainer(config).train()
    if not STAGE1_OUTPUT_DIR.exists():
        raise RuntimeError(f"Expected Stage 1 output not found at {STAGE1_OUTPUT_DIR}")
    return STAGE1_OUTPUT_DIR


def run_stage2(stage1_model_path: Path):
    """Run Stage 2 (full-parameter fine-tune, save, and HF push) initialized from `stage1_model_path`."""
    logger.info("=== Stage 2: full-parameter fine-tune (Qwen3-8B) ===")
    config = FullFineTuneConfig(
        stage1_model_path=stage1_model_path,
        output_dir=STAGE2_OUTPUT_DIR,
        hf_repo_id=HF_REPO_ID,
    )
    FullFineTuneTrainer(config).train()


def run_eval():
    """Run Recall@K/NDCG@K eval against Stage 2's saved model and write results to disk."""
    gc.collect()
    torch.cuda.empty_cache()

    logger.info("=== Eval: Recall@K/NDCG@K on the fine-tuned model ===")
    results = evaluate_ranking_metrics.run(STAGE2_OUTPUT_DIR, PROJECT_ROOT)
    formatted = evaluate_ranking_metrics.format_results(results)
    logger.info("\n%s", formatted)

    results_path = STAGE2_OUTPUT_DIR / "eval_results.txt"
    results_path.write_text(formatted, encoding="utf-8")
    logger.info("Saved eval results to %s", results_path)


if __name__ == "__main__":
    stage1_output = run_stage1()
    run_stage2(stage1_output)
    run_eval()
    logger.info("Full 8B retrain complete. Final model at %s", STAGE2_OUTPUT_DIR)
