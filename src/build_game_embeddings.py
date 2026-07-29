"""Orchestrates the item-embedding pipeline end to end.

Tokenizes the cleaned game catalog (tokenize_items) and generates item
embeddings from those tokens (embed_items), producing the final dataset
used downstream for RQ-VAE / semantic ID training.
"""

import time
from pathlib import Path

from embed_items import embed_items
from logger import Logger
from tokenize_items import tokenize_items

DATA_DIR = Path("data")

logger = Logger.get_logger(__name__)


class GameEmbeddingPipeline:
    """Two-stage pipeline: tokenize the catalog, then embed the tokens."""

    def __init__(
        self,
        input_path: Path = None,
        tokenized_path: Path = None,
        output_path: Path = None,
        limit: int = None,
    ):
        """Configure file paths and an optional item-count limit.

        Args:
            input_path: Source catalog parquet. Defaults to DATA_DIR/clean_game_catalog.parquet.
            tokenized_path: Pretokenized .npz destination. Defaults to DATA_DIR/tokenized_game_catalog.npz.
            output_path: Embeddings parquet destination. Defaults to DATA_DIR/output/games_with_embeddings.parquet.
            limit: If set, only process the first `limit` items.
        """
        self.input_path = input_path or DATA_DIR / "clean_game_catalog.parquet"
        self.tokenized_path = tokenized_path or DATA_DIR / "tokenized_game_catalog.npz"
        self.output_path = output_path or DATA_DIR / "output" / "games_with_embeddings.parquet"
        self.limit = limit

    def run(self):
        """Run tokenization followed by embedding generation; return the resulting DataFrame."""
        logger.info(
            "Starting embedding pipeline (input=%s, tokenized=%s, output=%s, limit=%s)",
            self.input_path, self.tokenized_path, self.output_path, self.limit,
        )
        pipeline_start = time.perf_counter()

        logger.info("[1/2] Tokenizing catalog: %s", self.input_path)
        stage_start = time.perf_counter()
        tokenize_items(
            input_path=self.input_path,
            output_path=self.tokenized_path,
            limit=self.limit,
        )
        logger.info("[1/2] Tokenization finished in %.1fs", time.perf_counter() - stage_start)

        logger.info("[2/2] Generating embeddings -> %s", self.output_path)
        stage_start = time.perf_counter()
        result_df = embed_items(
            input_path=self.input_path,
            output_path=self.output_path,
            tokenized_path=self.tokenized_path,
            limit=self.limit,
        )
        logger.info("[2/2] Embedding generation finished in %.1fs", time.perf_counter() - stage_start)

        logger.info(
            "Pipeline complete: %d items embedded, saved to %s (total %.1fs)",
            len(result_df), self.output_path, time.perf_counter() - pipeline_start,
        )
        return result_df


if __name__ == "__main__":
    GameEmbeddingPipeline().run()