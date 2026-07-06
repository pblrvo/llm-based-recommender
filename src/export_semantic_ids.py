"""Encodes item embeddings into semantic IDs with a trained RQ-VAE checkpoint,
and saves an id -> semantic_id mapping for the LLM fine-tuning stage.

Only the `id` and `embedding` columns are read from the embeddings parquet;
no other item metadata is needed for this step.
"""

from pathlib import Path

import numpy as np
import polars as pl
import torch
from tqdm import tqdm

from config import RQVAEConfig
from logger import Logger
from rqvae import RQVAE
from train_rqvae import get_device

logger = Logger.get_logger(__name__)


class SemanticIdExporter:
    def __init__(
        self,
        config: RQVAEConfig,
        checkpoint_path: Path,
        output_path: Path = None,
        device: str = None,
        batch_size: int = None,
    ):
        self.config = config
        self.checkpoint_path = Path(checkpoint_path)
        self.output_path = Path(output_path) if output_path else config.data_dir / "output" / "semantic_ids.parquet"
        self.device = device or get_device()
        self.batch_size = batch_size or config.batch_size

    def load_model(self) -> RQVAE:
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found at {self.checkpoint_path}. Train the model first with train_rqvae.py."
            )

        model = RQVAE(self.config).to(self.device)
        logger.info("Loading checkpoint from %s", self.checkpoint_path)
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        logger.info(
            "Loaded checkpoint: step=%s, best_val_loss=%s",
            checkpoint.get("global_step"), checkpoint.get("best_val_loss"),
        )
        return model

    def load_items(self) -> tuple:
        path = self.config.embeddings_path
        logger.info("Loading id + embedding columns from %s", path)
        df = pl.read_parquet(path, columns=["id", "embedding"])
        ids = df["id"].to_list()
        embeddings = np.array(df["embedding"].to_list(), dtype=np.float32)
        logger.info("Loaded %d items, embedding dim=%d", len(ids), embeddings.shape[1])

        if embeddings.shape[1] != self.config.item_embedding_dim:
            raise ValueError(
                f"Embeddings have dim {embeddings.shape[1]}, but config.item_embedding_dim="
                f"{self.config.item_embedding_dim}"
            )

        return ids, torch.from_numpy(embeddings)

    @torch.no_grad()
    def encode_all(self, model: RQVAE, embeddings: torch.Tensor) -> np.ndarray:
        n = embeddings.shape[0]
        all_semantic_ids = []
        for start in tqdm(range(0, n, self.batch_size), desc="Encoding semantic IDs"):
            batch = embeddings[start : start + self.batch_size].to(self.device)
            semantic_ids = model.encode_to_semantic_ids(batch)
            all_semantic_ids.append(semantic_ids.cpu())
        return torch.cat(all_semantic_ids, dim=0).numpy()

    def report_collisions(self, semantic_ids: np.ndarray) -> int:
        """Log how many items share an identical semantic ID with at least one
        other item. A collision means an LLM trained on these IDs cannot tell
        the affected items apart without extra disambiguation."""
        n_items = semantic_ids.shape[0]
        _, counts = np.unique(semantic_ids, axis=0, return_counts=True)
        n_unique_items = (counts == 1).sum()
        n_colliding_groups = (counts > 1).sum()
        n_colliding_items = n_items - n_unique_items

        if n_colliding_items > 0:
            logger.warning(
                "%d/%d items (%.2f%%) share a semantic ID with at least one other item, "
                "across %d colliding groups",
                n_colliding_items, n_items, 100 * n_colliding_items / n_items, n_colliding_groups,
            )
        else:
            logger.info("All %d items have a unique semantic ID", n_items)

        return n_colliding_items

    def export(self) -> pl.DataFrame:
        model = self.load_model()
        ids, embeddings = self.load_items()
        semantic_ids = self.encode_all(model, embeddings)

        self.report_collisions(semantic_ids)

        result_df = pl.DataFrame({"id": ids, "semantic_ids": semantic_ids.tolist()})
        for level in range(semantic_ids.shape[1]):
            result_df = result_df.with_columns(pl.Series(f"semantic_id_{level}", semantic_ids[:, level]))

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        result_df.write_parquet(self.output_path)
        logger.info("Saved %d id -> semantic_id mappings to %s", len(result_df), self.output_path)

        return result_df


if __name__ == "__main__":
    config = RQVAEConfig()
    checkpoint_path = config.checkpoint_dir / "rqvae_best.pt"
    SemanticIdExporter(config, checkpoint_path=checkpoint_path).export()
