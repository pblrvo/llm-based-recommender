"""Smoke test for the tokenize_items -> embed_items pipeline.

Runs both stages on a small sample of the cleaned game catalog and checks
that the resulting embeddings are well-formed (right shape, unit-normalized,
finite, and distinct across different games).
"""

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import polars as pl

from embed_items import EMBED_DIM, embed_items
from logger import Logger
from tokenize_items import tokenize_items

DATA_DIR = Path("data")
CATALOG_PATH = DATA_DIR / "clean_game_catalog.parquet"
SAMPLE_SIZE = 8
TEST_MAX_LENGTH = 256

logger = Logger.get_logger(__name__)


class EmbeddingGenerationTest:
    def __init__(self, sample_size: int = SAMPLE_SIZE, max_length: int = TEST_MAX_LENGTH):
        self.sample_size = sample_size
        self.max_length = max_length

    def run(self) -> pl.DataFrame:
        with TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)
            sample_catalog_path = tmp_dir / "sample_catalog.parquet"
            tokenized_path = tmp_dir / "sample_tokenized.npz"
            output_path = tmp_dir / "sample_with_embeddings.parquet"

            self._write_sample_catalog(sample_catalog_path)

            logger.info("Tokenizing %d sample items (max_length=%d)...", self.sample_size, self.max_length)
            tokenize_items(
                input_path=sample_catalog_path,
                output_path=tokenized_path,
                max_length=self.max_length,
            )
            self._check_tokenized_output(tokenized_path)
            logger.info("Tokenized output validated")

            logger.info("Generating embeddings for sample items...")
            result_df = embed_items(
                input_path=sample_catalog_path,
                output_path=output_path,
                tokenized_path=tokenized_path,
            )
            self._check_embeddings(result_df)

            logger.info("OK: generated %d embeddings of dimension %d.", self.sample_size, EMBED_DIM)
            return result_df

    def _write_sample_catalog(self, sample_catalog_path: Path):
        if not CATALOG_PATH.exists():
            logger.error("Catalog not found at %s", CATALOG_PATH)
            raise FileNotFoundError(
                f"{CATALOG_PATH} not found. Run notebooks/preprocess_australian_data.ipynb first."
            )
        catalog = pl.read_parquet(CATALOG_PATH)
        catalog.head(self.sample_size).write_parquet(sample_catalog_path)
        logger.debug("Wrote %d-row sample catalog to %s", self.sample_size, sample_catalog_path)

    def _check_tokenized_output(self, tokenized_path: Path):
        data = np.load(tokenized_path)
        assert set(data.keys()) >= {"input_ids", "attention_mask", "n_items"}, (
            f"Unexpected keys in tokenized output: {list(data.keys())}"
        )
        assert data["input_ids"].shape == (self.sample_size, self.max_length)
        assert data["attention_mask"].shape == (self.sample_size, self.max_length)
        assert int(data["n_items"]) == self.sample_size
        logger.debug(
            "Tokenized output shape OK: input_ids=%s, attention_mask=%s",
            data["input_ids"].shape, data["attention_mask"].shape,
        )

    def _check_embeddings(self, result_df: pl.DataFrame):
        assert "embedding" in result_df.columns
        assert len(result_df) == self.sample_size

        embeddings = np.array(result_df["embedding"].to_list())
        assert embeddings.shape == (self.sample_size, EMBED_DIM)
        assert np.isfinite(embeddings).all(), "Embeddings contain NaN/Inf values"
        logger.debug("Embeddings shape OK: %s, all finite", embeddings.shape)

        # atol is loose because the model runs in bfloat16 (~3 significant digits),
        # so the post-normalization cast to float32 won't land exactly on 1.0.
        norms = np.linalg.norm(embeddings, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-2), f"Embeddings are not L2-normalized: {norms}"
        logger.debug("L2 norms OK: min=%.4f, max=%.4f", norms.min(), norms.max())

        # Different games should not collapse to (near-)identical vectors.
        pairwise_sim = embeddings @ embeddings.T
        off_diagonal = pairwise_sim[~np.eye(self.sample_size, dtype=bool)]
        assert off_diagonal.max() < 0.999, "Distinct items produced near-identical embeddings"
        logger.debug("Pairwise similarity OK: max off-diagonal=%.4f", off_diagonal.max())


if __name__ == "__main__":
    EmbeddingGenerationTest().run()
