"""Trains the RQ-VAE model on item embeddings.

Loads embeddings produced by embed_items.py, trains RQVAE with a
warmup+cosine learning-rate schedule and periodic dead-codebook resets,
and logs metrics to TensorBoard (run `tensorboard --logdir runs` to view).
"""

import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from config import RQVAEConfig
from logger import Logger
from lr_scheduler import WarmupCosineScheduler
from rqvae import RQVAE

logger = Logger.get_logger(__name__)


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class RQVAETrainer:
    def __init__(
        self,
        config: RQVAEConfig,
        device: str = None,
        tensorboard_dir: Path = None,
        val_every: int = 1,
        checkpoint_every: int = 500,
        histogram_every: int = 100,
    ):
        self.config = config
        self.device = device or get_device()
        self.tensorboard_dir = Path(tensorboard_dir) if tensorboard_dir else (
            Path("runs") / f"rqvae_{time.strftime('%Y%m%d_%H%M%S')}"
        )
        self.val_every = val_every
        self.checkpoint_every = checkpoint_every
        self.histogram_every = histogram_every

        self.model: RQVAE = None
        self.optimizer: torch.optim.Optimizer = None
        self.scheduler: WarmupCosineScheduler = None
        self.writer: SummaryWriter = None
        self.train_loader: DataLoader = None
        self.val_loader: DataLoader = None
        self.global_step = 0
        self.best_val_loss = float("inf")

        logger.info("RQVAETrainer using device: %s", self.device)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def load_data(self):
        path = self.config.embeddings_path
        logger.info("Loading item embeddings from %s", path)
        df = pl.read_parquet(path, columns=["embedding"])
        embeddings = np.array(df["embedding"].to_list(), dtype=np.float32)
        logger.info("Loaded %d embeddings of dim %d", embeddings.shape[0], embeddings.shape[1])

        if embeddings.shape[1] != self.config.item_embedding_dim:
            raise ValueError(
                f"Embeddings have dim {embeddings.shape[1]}, but config.item_embedding_dim="
                f"{self.config.item_embedding_dim}"
            )

        tensor = torch.from_numpy(embeddings)
        n = tensor.shape[0]
        n_val = max(1, int(n * self.config.val_split))

        perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
        val_tensor = tensor[perm[:n_val]]
        train_tensor = tensor[perm[n_val:]]
        logger.info("Train/val split: %d train, %d val (val_split=%.2f)", len(train_tensor), len(val_tensor), self.config.val_split)

        self.train_loader = DataLoader(TensorDataset(train_tensor), batch_size=self.config.batch_size, shuffle=True)
        self.val_loader = DataLoader(TensorDataset(val_tensor), batch_size=self.config.batch_size, shuffle=False)

    def build(self):
        self.model = RQVAE(self.config).to(self.device)

        if self.config.use_kmeans_init:
            self.model.kmeans_init(self.train_loader, self.device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.max_lr)

        steps_per_epoch = len(self.train_loader)
        total_steps = steps_per_epoch * self.config.num_epochs
        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            total_steps=total_steps,
            max_lr=self.config.max_lr,
            min_lr=self.config.min_lr,
            warmup_steps=self.config.warmup_steps,
            warmup_start_lr=self.config.warmup_start_lr,
            scheduler_type=self.config.scheduler_type,
        )

        self.tensorboard_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.tensorboard_dir))
        logger.info("TensorBoard logging to %s (steps/epoch=%d, total_steps=%d)", self.tensorboard_dir, steps_per_epoch, total_steps)

    # ------------------------------------------------------------------
    # Train / validate
    # ------------------------------------------------------------------

    def train_step(self, batch: torch.Tensor) -> dict:
        self.model.train()
        batch = batch.to(self.device)

        self.optimizer.zero_grad()
        _, all_indices, loss_dict = self.model(batch)
        loss_dict["loss"].backward()

        # Always compute the grad norm (even with clipping off) so it can be logged.
        max_norm = self.config.gradient_clip_norm if self.config.use_gradient_clipping else float("inf")
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)

        self.optimizer.step()
        lr = self.scheduler.step()
        self.global_step += 1

        metrics = self._extract_metrics(loss_dict, all_indices)
        metrics["grad_norm"] = grad_norm.item()
        metrics["lr"] = lr

        if self.config.reset_unused_codes and self.global_step % self.config.steps_per_codebook_reset == 0:
            self._maybe_reset_dead_codes(loss_dict["level_residuals"])

        return metrics

    def _maybe_reset_dead_codes(self, level_residuals: list):
        for level, vq_layer in enumerate(self.model.vq_layers):
            usage_rate = vq_layer.get_usage_rate()
            max_share = vq_layer.get_max_usage_share()
            if usage_rate < self.config.codebook_usage_threshold or max_share > self.config.codebook_dominance_threshold:
                vq_layer.reset_unused_codebook_vectors(
                    level_residuals[level], dominance_threshold=self.config.codebook_dominance_threshold
                )

    @torch.no_grad()
    def validate(self) -> dict:
        self.model.eval()
        total_loss, total_recon, total_vq, n_batches = 0.0, 0.0, 0.0, 0
        all_semantic_ids = []

        for (batch,) in self.val_loader:
            batch = batch.to(self.device)
            _, all_indices, loss_dict = self.model(batch)
            total_loss += loss_dict["loss"].item()
            total_recon += loss_dict["recon_loss"].item()
            total_vq += self._scalar(loss_dict["vq_loss"])
            n_batches += 1
            all_semantic_ids.append(torch.stack(all_indices, dim=-1))

        semantic_ids = torch.cat(all_semantic_ids, dim=0)
        return {
            "loss": total_loss / n_batches,
            "recon_loss": total_recon / n_batches,
            "vq_loss": total_vq / n_batches,
            "unique_ids_proportion": self.model.calculate_unique_ids_proportion(semantic_ids),
        }

    def _extract_metrics(self, loss_dict: dict, all_indices: list) -> dict:
        semantic_ids = torch.stack(all_indices, dim=-1)
        return {
            "loss": loss_dict["loss"].item(),
            "recon_loss": loss_dict["recon_loss"].item(),
            "vq_loss": self._scalar(loss_dict["vq_loss"]),
            "codebook_losses": [l.item() for l in loss_dict["codebook_losses"]],
            "commitment_losses": [l.item() for l in loss_dict["commitment_losses"]],
            "codebook_usage": self.model.calculate_codebook_usage(),
            "codebook_max_share": self.model.calculate_codebook_max_share(),
            "unique_ids_proportion": self.model.calculate_unique_ids_proportion(semantic_ids),
            "avg_residual_norm": self.model.calculate_avg_residual_norm(loss_dict["residual"]),
        }

    @staticmethod
    def _scalar(value) -> float:
        return value.item() if isinstance(value, torch.Tensor) else float(value)

    # ------------------------------------------------------------------
    # Logging / checkpointing
    # ------------------------------------------------------------------

    def log_metrics(self, metrics: dict, prefix: str):
        w = self.writer
        w.add_scalar(f"{prefix}/loss", metrics["loss"], self.global_step)
        w.add_scalar(f"{prefix}/recon_loss", metrics["recon_loss"], self.global_step)
        w.add_scalar(f"{prefix}/vq_loss", metrics["vq_loss"], self.global_step)
        w.add_scalar(f"{prefix}/unique_ids_proportion", metrics["unique_ids_proportion"], self.global_step)

        if "avg_residual_norm" in metrics:
            w.add_scalar(f"{prefix}/avg_residual_norm", metrics["avg_residual_norm"], self.global_step)

        for level, cb_loss in enumerate(metrics.get("codebook_losses", [])):
            w.add_scalar(f"{prefix}/codebook_loss/level_{level}", cb_loss, self.global_step)
        for level, cm_loss in enumerate(metrics.get("commitment_losses", [])):
            w.add_scalar(f"{prefix}/commitment_loss/level_{level}", cm_loss, self.global_step)
        for level, usage in enumerate(metrics.get("codebook_usage", [])):
            w.add_scalar(f"{prefix}/codebook_usage/level_{level}", usage, self.global_step)
        for level, max_share in enumerate(metrics.get("codebook_max_share", [])):
            w.add_scalar(f"{prefix}/codebook_max_share/level_{level}", max_share, self.global_step)

        if "grad_norm" in metrics:
            w.add_scalar("train/grad_norm", metrics["grad_norm"], self.global_step)
        if "lr" in metrics:
            w.add_scalar("train/lr", metrics["lr"], self.global_step)

    def log_histograms(self):
        for level, vq_layer in enumerate(self.model.vq_layers):
            self.writer.add_histogram(f"codebook_weights/level_{level}", vq_layer.embedding.weight, self.global_step)
        for name, param in self.model.encoder.named_parameters():
            self.writer.add_histogram(f"encoder_weights/{name}", param, self.global_step)

    def save_checkpoint(self, tag: str):
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.config.checkpoint_dir / f"rqvae_{tag}.pt"
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "global_step": self.global_step,
                "best_val_loss": self.best_val_loss,
                "config": self.config,
            },
            path,
        )
        logger.info("Saved checkpoint: %s (step=%d)", path, self.global_step)

    def load_checkpoint(self, path: Path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint["best_val_loss"]
        logger.info("Loaded checkpoint from %s (step=%d)", path, self.global_step)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def train(self):
        self.load_data()
        self.build()

        steps_per_epoch = len(self.train_loader)
        total_steps = steps_per_epoch * self.config.num_epochs
        logger.info("Starting training: %d epochs, %d steps/epoch, %d total steps", self.config.num_epochs, steps_per_epoch, total_steps)

        train_start = time.perf_counter()
        for epoch in range(self.config.num_epochs):
            epoch_start = time.perf_counter()
            last_train_metrics = None

            for (batch,) in self.train_loader:
                last_train_metrics = self.train_step(batch)
                self.log_metrics(last_train_metrics, prefix="train")

            val_metrics = None
            if (epoch + 1) % self.val_every == 0:
                val_metrics = self.validate()
                self.log_metrics(val_metrics, prefix="val")
                if val_metrics["loss"] < self.best_val_loss:
                    self.best_val_loss = val_metrics["loss"]
                    self.save_checkpoint(tag="best")

            if (epoch + 1) % self.checkpoint_every == 0:
                self.save_checkpoint(tag=f"epoch_{epoch + 1}")

            if (epoch + 1) % self.histogram_every == 0:
                self.log_histograms()

            val_summary = f"val_loss={val_metrics['loss']:.4f}" if val_metrics else "val_loss=—"
            logger.info(
                "Epoch %d/%d: train_loss=%.4f %s lr=%.2e grad_norm=%.3f unique_ids=%.2f [%.1fs]",
                epoch + 1, self.config.num_epochs, last_train_metrics["loss"], val_summary,
                last_train_metrics["lr"], last_train_metrics["grad_norm"],
                last_train_metrics["unique_ids_proportion"], time.perf_counter() - epoch_start,
            )

        self.save_checkpoint(tag="final")
        logger.info("Training complete in %.1fs (best_val_loss=%.4f)", time.perf_counter() - train_start, self.best_val_loss)
        self.writer.close()


if __name__ == "__main__":
    config = RQVAEConfig()
    trainer = RQVAETrainer(config)
    trainer.train()
