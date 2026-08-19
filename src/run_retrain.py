"""Orchestrates the full Stage 1 (embedding warmup) + Stage 2 (QLoRA)
retrain for the Qwen3-4B model, writing to fresh outputs/ directories.
"""

from pathlib import Path

from logger import Logger
from qlora_finetune import QLoraFineTuneConfig, QLoraFineTuneTrainer
from warmup_embeddings import EmbeddingWarmupConfig, EmbeddingWarmupTrainer

logger = Logger.get_logger(__name__)

STAGE1_OUTPUT_DIR = Path("outputs/qwen3-4b-embed-warmup")
STAGE1_MAX_STEPS = 750
STAGE1_SAVE_STEPS = 250
STAGE2_OUTPUT_DIR = Path("outputs/qwen3-4b-qlora")


def run_stage1():
    """Run Stage 1 (embedding warmup) with the Qwen3-4B / 4-bit configuration and return its checkpoint path."""
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
    """Run Stage 2 (QLoRA fine-tune) initialized from `stage1_checkpoint`."""
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