"""Smoke test for the tokenize_items -> embed_items pipeline.

Runs both stages on a small sample of the cleaned game catalog and checks
that the resulting embeddings are well-formed (right shape, unit-normalized,
finite, and distinct across different games). Marked `slow` since it loads
a real embedding model and needs the real catalog on disk -- skipped
automatically if that catalog isn't present.
"""

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import polars as pl
import pytest

from embed_items import EMBED_DIM, embed_items
from tokenize_items import tokenize_items

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CATALOG_PATH = DATA_DIR / "clean_game_catalog.parquet"
SAMPLE_SIZE = 8
TEST_MAX_LENGTH = 256


@pytest.mark.slow
@pytest.mark.skipif(
    not CATALOG_PATH.exists(),
    reason=f"{CATALOG_PATH} not found -- run notebooks/preprocess_australian_data.ipynb first",
)
def test_tokenize_and_embed_produce_valid_embeddings():
    with TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        sample_catalog_path = tmp_dir / "sample_catalog.parquet"
        tokenized_path = tmp_dir / "sample_tokenized.npz"
        output_path = tmp_dir / "sample_with_embeddings.parquet"

        catalog = pl.read_parquet(CATALOG_PATH)
        catalog.head(SAMPLE_SIZE).write_parquet(sample_catalog_path)

        tokenize_items(input_path=sample_catalog_path, output_path=tokenized_path, max_length=TEST_MAX_LENGTH)

        # np.load() on an .npz keeps the underlying zip file handle open
        # until closed -- on Windows, TemporaryDirectory's cleanup then
        # fails with PermissionError if that handle outlives the `with`
        # block, so this must be its own nested context manager.
        with np.load(tokenized_path) as data:
            assert set(data.keys()) >= {"input_ids", "attention_mask", "n_items"}
            assert data["input_ids"].shape == (SAMPLE_SIZE, TEST_MAX_LENGTH)
            assert data["attention_mask"].shape == (SAMPLE_SIZE, TEST_MAX_LENGTH)
            assert int(data["n_items"]) == SAMPLE_SIZE

        result_df = embed_items(
            input_path=sample_catalog_path, output_path=output_path, tokenized_path=tokenized_path,
        )

        assert "embedding" in result_df.columns
        assert len(result_df) == SAMPLE_SIZE

        embeddings = np.array(result_df["embedding"].to_list())
        assert embeddings.shape == (SAMPLE_SIZE, EMBED_DIM)
        assert np.isfinite(embeddings).all(), "Embeddings contain NaN/Inf values"

        # atol is loose because the model runs in bfloat16 (~3 significant
        # digits), so the post-normalization cast to float32 won't land
        # exactly on 1.0.
        norms = np.linalg.norm(embeddings, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-2), f"Embeddings are not L2-normalized: {norms}"

        # Different games should not collapse to (near-)identical vectors.
        pairwise_sim = embeddings @ embeddings.T
        off_diagonal = pairwise_sim[~np.eye(SAMPLE_SIZE, dtype=bool)]
        assert off_diagonal.max() < 0.999, "Distinct items produced near-identical embeddings"
