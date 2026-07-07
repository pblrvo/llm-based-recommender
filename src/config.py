from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from logger import Logger

logger = Logger.get_logger(__name__)


@dataclass
class RQVAEConfig:
    data_dir: Path = Path("data")
    embeddings_path: Optional[Path] = None
    checkpoint_dir: Path = Path("checkpoints")

    # Model parameters
    item_embedding_dim: int = 1024  # Input embedding dimension (e.g., Qwen3-0.6B)
    encoder_hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])  # Encoder layers
    codebook_embedding_dim: int = 32  # Dimension of codebook vectors
    codebook_quantization_levels: int = 3  # Number of hierarchical levels
    codebook_size: int = 256  # Number of codes per codebook
    codebook_normalize: bool = False  # L2-normalize encoder output before quantization
    commitment_weight: float = 0.1  # Commitment loss weight (beta)

    # Training parameters
    batch_size: int = 32768  # Batch size for training
    gradient_accumulation_steps: int = 1  # Number of gradient accumulation steps
    num_epochs: int = 20000  # Number of training epochs
    scheduler_type: str = "cosine_with_warmup"  # Learning rate scheduler type ("cosine", "cosine_with_warmup")
    warmup_start_lr: float = 1e-8  # Starting learning rate for warmup (only for cosine_with_warmup)
    warmup_steps: int = 200  # Number of warmup steps (only for cosine_with_warmup)
    max_lr: float = 3e-4  # Maximum learning rate (start of cosine)
    min_lr: float = 1e-6  # Minimum learning rate (end of cosine)
    use_gradient_clipping: bool = True  # Enable gradient clipping
    gradient_clip_norm: float = 1.0  # Maximum gradient norm for clipping
    use_kmeans_init: bool = True  # Use k-means initialization for codebooks
    reset_unused_codes: bool = True  # Reset unused codebook codes during training
    steps_per_codebook_reset: int = 2  # Reset unused codebook codes every N steps (breaks if set to 1)
    codebook_usage_threshold: float = 1.0  # Only reset if usage falls below this proportion (0-1)
    codebook_dominance_threshold: float = 0.5  # Also reset a code if it claims more than this share of usage
    val_split: float = 0.05  # Validation set split ratio

    def __post_init__(self):
        """Validate configuration and set computed fields."""
        # Auto-generate embeddings path if not provided
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
