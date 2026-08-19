"""Configuration dataclass for the RQ-VAE training pipeline."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from logger import Logger

logger = Logger.get_logger(__name__)


@dataclass
class RQVAEConfig:
    """Configuration for training and evaluating the RQ-VAE.

    Holds file paths, model architecture hyperparameters, and training-loop
    settings. Defaults reflect this project's standard run (Qwen3-0.6B
    embeddings, 3 hierarchical quantization levels, codebook size 256).
    """

    data_dir: Path = Path("data")
    embeddings_path: Optional[Path] = None
    checkpoint_dir: Path = Path("checkpoints")

    # Model parameters
    item_embedding_dim: int = 1024
    encoder_hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
    codebook_embedding_dim: int = 32
    codebook_quantization_levels: int = 3
    codebook_size: int = 256
    codebook_normalize: bool = False
    commitment_weight: float = 0.1  # beta

    # Training parameters
    batch_size: int = 32768
    gradient_accumulation_steps: int = 1
    num_epochs: int = 20000
    scheduler_type: str = "cosine_with_warmup"  # "cosine" or "cosine_with_warmup"
    warmup_start_lr: float = 1e-8  # only for cosine_with_warmup
    warmup_steps: int = 200  # only for cosine_with_warmup
    max_lr: float = 3e-4
    min_lr: float = 1e-6
    use_gradient_clipping: bool = True
    gradient_clip_norm: float = 1.0
    use_kmeans_init: bool = True
    reset_unused_codes: bool = True
    steps_per_codebook_reset: int = 2  # breaks if set to 1
    codebook_usage_threshold: float = 1.0  # 0-1
    codebook_dominance_threshold: float = 0.5
    val_split: float = 0.05

    def __post_init__(self):
        """Validate configuration, fill in computed defaults, and log a summary."""
        if self.embeddings_path is None:
            self.embeddings_path = self.data_dir / "output" / "games_with_embeddings.parquet"
            logger.info("embeddings_path not set, defaulting to %s", self.embeddings_path)

        if self.scheduler_type not in ("cosine", "cosine_with_warmup"):
            raise ValueError(f"Unknown scheduler_type: {self.scheduler_type!r}")

        logger.info(
            "RQVAEConfig: item_dim=%d, encoder_hidden_dims=%s, codebook_dim=%d, "
            "levels=%d, codebook_size=%d, normalize=%s, commitment_weight=%.3f",
            self.item_embedding_dim, self.encoder_hidden_dims, self.codebook_embedding_dim,
            self.codebook_quantization_levels, self.codebook_size, self.codebook_normalize,
            self.commitment_weight,
        )
        logger.info(
            "RQVAEConfig: batch_size=%d, num_epochs=%d, scheduler=%s, max_lr=%.2e, min_lr=%.2e, "
            "warmup_steps=%d",
            self.batch_size, self.num_epochs, self.scheduler_type, self.max_lr, self.min_lr,
            self.warmup_steps,
        )
        logger.info(
            "RQVAEConfig: reset_unused_codes=%s, steps_per_codebook_reset=%d, "
            "codebook_usage_threshold=%.2f, codebook_dominance_threshold=%.2f",
            self.reset_unused_codes, self.steps_per_codebook_reset,
            self.codebook_usage_threshold, self.codebook_dominance_threshold,
        )