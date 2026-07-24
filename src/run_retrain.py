"""Orchestrates the full Stage 1 (embedding warmup) + Stage 2 (QLoRA)
retrain on the rebalanced dataset (grounding train/val fix, NL query tasks,
cross-task exposure cap -- see build_finetune_dataset.py), for the Qwen3-4B
pivot on this project's 12GB GPU.

Both stages' own __main__ blocks default to stale configs (warmup_embeddings
defaults to the old Qwen3-0.6B/non-quantized path; qlora_finetune's
stage1_adapter_path assumes a specific prior checkpoint number) -- this
script is the actual 4B invocation, explicit about every override instead
of relying on either file's defaults.

Writes to fresh outputs/ directories, not models/qwen3-4b-qlora (the
already-HF-uploaded prior run) -- that directory is left untouched so it
stays available as a fallback until the new run's eval results are reviewed
and it's deliberately promoted.
"""

from pathlib import Path

from logger import Logger
from qlora_finetune import QLoraFineTuneConfig, QLoraFineTuneTrainer
from warmup_embeddings import EmbeddingWarmupConfig, EmbeddingWarmupTrainer

logger = Logger.get_logger(__name__)

STAGE1_OUTPUT_DIR = Path("outputs/qwen3-4b-embed-warmup")
STAGE1_MAX_STEPS = 750
STAGE1_SAVE_STEPS = 250  # checkpoints at 250/500/750 -- final kept via save_total_limit=3
STAGE2_OUTPUT_DIR = Path("outputs/qwen3-4b-qlora")


def run_stage1():
    logger.info("=== Stage 1: embedding warmup (Qwen3-4B, 4-bit) ===")
    config = EmbeddingWarmupConfig(
        base_model="Qwen/Qwen3-4B",
        load_in_4bit=True,
        output_dir=STAGE1_OUTPUT_DIR,
        max_steps=STAGE1_MAX_STEPS,
        save_steps=STAGE1_SAVE_STEPS,
    )
    EmbeddingWarmupTrainer(config).train()
    checkpoint_path = STAGE1_OUTPUT_DIR / f"checkpoint-{STAGE1_MAX_STEPS}"
    if not checkpoint_path.exists():
        raise RuntimeError(f"Expected Stage 1 checkpoint not found at {checkpoint_path}")
    return checkpoint_path


def run_stage2(stage1_checkpoint: Path):
    logger.info("=== Stage 2: QLoRA fine-tune (Qwen3-4B) ===")
    config = QLoraFineTuneConfig(
        stage1_adapter_path=stage1_checkpoint,
        output_dir=STAGE2_OUTPUT_DIR,
    )
    QLoraFineTuneTrainer(config).train()


if __name__ == "__main__":
    stage1_checkpoint = run_stage1()
    run_stage2(stage1_checkpoint)
    logger.info("Full retrain complete. Final adapter at %s", STAGE2_OUTPUT_DIR)
