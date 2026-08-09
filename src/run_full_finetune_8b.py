"""Orchestrates the full Stage 1 (embedding warmup) + Stage 2 (full-parameter
fine-tune) + eval + Hugging Face Hub push run for the Qwen3-8B RunPod run --
the bigger-model, full-epoch alternative to run_retrain.py's Qwen3-4B/QLoRA
pipeline, intended for a RunPod GPU (80GB class) rather than this project's
original 12GB local card.

Both stages' own __main__ blocks default to the 4B/QLoRA pipeline's configs
-- this script is the actual 8B invocation, explicit about every override
instead of relying on either file's defaults.

Stage 1 uses warmup_embeddings.py's load_in_4bit=False path (8B fits
unquantized in bf16, ~16GB) -- its final save is a complete, directly-
loadable model (not a PEFT adapter, unlike the 4-bit path), which is what
full_finetune_8b.py's Stage 2 expects.

Writes to fresh outputs/ directories, matching run_retrain.py's convention.

Pipeline order: Stage 1 -> Stage 2 (trains, saves locally, pushes to
HF_REPO_ID) -> eval (Recall@K/NDCG@K on the just-saved model, written to
outputs/qwen3-8b-full-finetune/eval_results.txt). Eval runs AFTER the HF
push, not before -- the push is cheap and reversible (re-push after fixing
anything eval turns up), and keeping "train, save, push" as one contained
step in full_finetune_8b.py's train() avoids splitting that responsibility
across two files.

warmup_embeddings.py imports unsloth at module level (needed for its
load_in_4bit=True path), and unsloth monkey-patches transformers/trl at
import time for its whole process, not just call sites that use it. Hit in
practice: importing `run_stage2`/`run_eval` from this module (without ever
calling `run_stage1`) still triggered Unsloth's global patching via this
module's own top-level `from warmup_embeddings import ...`, which altered
trl's SFTConfig `eos_token` default and broke full_finetune_8b.py's plain
(non-Unsloth) SFTTrainer construction with `ValueError: The specified
eos_token ('<EOS_TOKEN>') is not found in the vocabulary`. The
`warmup_embeddings` import is local to run_stage1() below specifically so
that calling run_stage2()/run_eval() alone -- e.g. to resume Stage 2 after
Stage 1 already completed -- never imports unsloth at all.
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

# Set to None to skip the Hugging Face push (see FullFineTuneConfig.hf_repo_id
# -- requires HF_TOKEN or a prior `huggingface-cli login` on the machine
# running this, checked at the start of Stage 2 before training begins).
HF_REPO_ID = "pblrvo/Qwen3-8B-Game-semantic-IDs-v3"


def run_stage1():
    """Run Stage 1 (embedding warmup) with the Qwen3-8B / non-quantized configuration.

    Returns:
        Path to Stage 1's output directory (a complete model, not a numbered
        checkpoint -- the non-quantized path's final save_pretrained() is
        NOT a known-broken no-op, unlike the 4-bit path run_retrain.py uses).
    """
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
    """Run Recall@K/NDCG@K eval against Stage 2's saved model and write results to disk.

    Stage 2's model/trainer objects are local to run_stage2() and go out of
    scope once it returns, but PyTorch's CUDA caching allocator doesn't
    always release that memory back immediately -- gc.collect() +
    empty_cache() force it before eval loads its own (separate) copy of the
    model, so this doesn't OOM trying to hold two 8B models at once.
    """
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
