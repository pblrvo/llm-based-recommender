import os
import sys
import time
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Reduces allocator fragmentation from the widely varying batch sizes/shapes
# produced by length-bucketed batching (see PyTorch's own OOM error message).
# Not supported on Windows (CUDAAllocatorConfig warns and ignores it there).
if sys.platform != "win32":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm
from transformers import AutoModel

from logger import Logger

DATA_DIR = Path("data")

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
BATCH_SIZE = 64
EMBED_DIM = 1024

# Batch size is scaled down for longer sequences since attention memory grows
# roughly with sequence length squared; BATCH_SIZE is only safe up to
# REFERENCE_SEQ_LEN tokens.
REFERENCE_SEQ_LEN = 512

# How often (in seconds) to persist progress to disk. Embedding the full
# catalog takes hours; without checkpoints, a crash near the end (e.g. an
# OOM on the last few long-sequence batches) throws away everything.
CHECKPOINT_INTERVAL_SECONDS = 300

logger = Logger.get_logger(__name__)


def get_device() -> str:
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    logger.info("Using device: %s", device)
    return device


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """"Extracts embeddings using last token pooling"""
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
    

def generate_embeddings(
        model: AutoModel,
        device: str,
        pretokenized_batch: dict,
        target_dim = 1024,
) -> np.ndarray:

    # Move to device
    encoded = {k: v.to(device) for k, v in pretokenized_batch.items()}

    # Generate embeddings
    with torch.no_grad():
        outputs = model(**encoded)

        # Use last token pooling
        embeddings = last_token_pool(outputs.last_hidden_state, encoded["attention_mask"])

        # Truncate to target dimension if specified
        if target_dim and target_dim < embeddings.shape[1]:
            embeddings = embeddings[:, :target_dim]

        # L2 normalize
        embeddings = F.normalize(embeddings, p=2, dim=1)

    return embeddings.float().cpu().numpy()


def generate_embeddings_with_oom_retry(model, device, batch: dict, target_dim: int) -> np.ndarray:
    """Runs generate_embeddings, halving the batch and retrying on CUDA OOM.

    Safety net for cases where the adaptive batch size (calibrated on
    REFERENCE_SEQ_LEN) still doesn't fit for a particular batch/GPU.
    """
    batch_size = batch["input_ids"].size(0)
    try:
        return generate_embeddings(model, device, batch, target_dim)
    except torch.OutOfMemoryError:
        if batch_size == 1:
            raise
        seq_len = batch["input_ids"].size(1)
        logger.warning(
            "CUDA OOM at batch_size=%d, seq_len=%d; freeing cache and retrying as two smaller batches",
            batch_size, seq_len,
        )
        torch.cuda.empty_cache()
        mid = batch_size // 2
        first_half = {k: v[:mid] for k, v in batch.items()}
        second_half = {k: v[mid:] for k, v in batch.items()}
        return np.concatenate([
            generate_embeddings_with_oom_retry(model, device, first_half, target_dim),
            generate_embeddings_with_oom_retry(model, device, second_half, target_dim),
        ])


def adaptive_batch_size(seq_len: int, base_batch_size: int = BATCH_SIZE, reference_len: int = REFERENCE_SEQ_LEN) -> int:
    """Shrinks batch size for longer sequences (attention memory grows ~seq_len^2)."""
    if seq_len <= reference_len:
        return base_batch_size
    scale = (reference_len / seq_len) ** 2
    return max(1, int(base_batch_size * scale))


def _load_checkpoint(checkpoint_path: Path, total_items: int):
    if not checkpoint_path.exists():
        return None
    with np.load(checkpoint_path) as ckpt:
        if int(ckpt["n_items"]) != total_items:
            logger.warning(
                "Checkpoint item count (%d) doesn't match current catalog (%d); ignoring checkpoint",
                int(ckpt["n_items"]), total_items,
            )
            return None
        return ckpt["embeddings"].copy(), ckpt["filled"].copy()


def _save_checkpoint(checkpoint_path: Path, embeddings: np.ndarray, filled: np.ndarray, total_items: int):
    # Write to a temp file and rename, so a crash mid-write can't corrupt
    # the last good checkpoint.
    tmp_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    with open(tmp_path, "wb") as f:
        np.savez(f, embeddings=embeddings, filled=filled, n_items=total_items)
    tmp_path.replace(checkpoint_path)
    logger.info("Checkpoint saved: %d/%d items complete", int(filled.sum()), total_items)

def embed_items(input_path: Path = None, output_path: Path = None, tokenized_path: Path = None, limit: int = None):
    device = get_device()
    input_path = input_path or DATA_DIR / "clean_game_catalog.parquet"
    output_path = output_path or DATA_DIR / "output" / "games_with_embeddings.parquet"
    tokenized_path = tokenized_path or DATA_DIR / "tokenized_game_catalog.npz"

    # Load data
    logger.info("Loading item catalog from %s", input_path)
    item_df = pl.read_parquet(input_path)
    if limit:
        item_df = item_df.head(limit)
        logger.info("Limiting to first %d items", limit)
    total_items = len(item_df)
    pl.Config.set_fmt_str_lengths(2000)
    logger.info("Loaded %d items from catalog", total_items)

    logger.info("Loading model %s", MODEL_NAME)
    load_start = time.perf_counter()
    model = AutoModel.from_pretrained(MODEL_NAME)

    model = model.to(device)
    model.eval()
    logger.info("Model loaded and moved to %s in %.1fs", device, time.perf_counter() - load_start)

    use_compile = True
    if device == "cuda" and use_compile:
        try:
            import triton  # noqa: F401
        except ImportError:
            logger.warning("triton not installed; skipping torch.compile (falling back to eager mode)")
        else:
            logger.info("Compiling model with torch.compile")
            model = torch.compile(model)

    # Load pre-tokenized data
    if not tokenized_path.exists():
        logger.error("Pre-tokenized data not found at %s", tokenized_path)
        raise FileNotFoundError(
            f"Pre-tokenized data not found at {tokenized_path}. Please run src/tokenize_items.py first."
        )

    logger.info("Loading pre-tokenized data from %s", tokenized_path)
    with np.load(tokenized_path) as pretokenized_data:
        # Verify data matches
        if pretokenized_data["n_items"] != total_items:
            logger.error(
                "Item count mismatch: tokenized=%d, catalog=%d",
                pretokenized_data["n_items"], total_items,
            )
            raise ValueError(
                f"Pre-tokenized data has {pretokenized_data['n_items']} items, but current data has {total_items}"
            )

        input_ids = torch.from_numpy(pretokenized_data["input_ids"])
        attention_mask = torch.from_numpy(pretokenized_data["attention_mask"])
    logger.info("Pre-tokenized data validated: %d items, padded sequence length %d", total_items, input_ids.shape[1])

    # Real per-item token counts (excluding padding). Padded length is a fixed
    # worst case (e.g. 2000), but most items are much shorter (see notebook
    # EDA); since attention cost scales ~quadratically with sequence length,
    # batching items of similar real length together and trimming each batch
    # down to only what it needs avoids paying for padding on every batch.
    real_lengths = attention_mask.sum(dim=1)
    length_percentiles = torch.quantile(real_lengths.float(), torch.tensor([0.5, 0.9, 0.99]))
    logger.info(
        "Real token length: median=%d p90=%d p99=%d max=%d (padded to %d)",
        length_percentiles[0], length_percentiles[1], length_percentiles[2],
        real_lengths.max(), input_ids.shape[1],
    )

    # Resume from a checkpoint if one exists (e.g. a previous run crashed).
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_path.parent / f"{output_path.stem}.checkpoint.npz"
    checkpoint = _load_checkpoint(checkpoint_path, total_items)
    if checkpoint is not None:
        all_embeddings, filled = checkpoint
        logger.info("Resuming from checkpoint: %d/%d items already embedded", int(filled.sum()), total_items)
    else:
        all_embeddings = np.zeros((total_items, EMBED_DIM), dtype=np.float32)
        filled = np.zeros(total_items, dtype=bool)

    remaining_positions = torch.from_numpy(np.flatnonzero(~filled))
    length_sorted_order = remaining_positions[torch.argsort(real_lengths[remaining_positions])]

    logger.info("Generating embeddings: base_batch_size=%d, embed_dim=%d", BATCH_SIZE, EMBED_DIM)

    embed_start = time.perf_counter()
    last_checkpoint_time = embed_start
    with tqdm(total=total_items, initial=int(filled.sum()), desc="Generating Embeddings") as progress_bar:
        position = 0
        batch_num = 0
        while position < len(length_sorted_order):
            current_len = int(real_lengths[length_sorted_order[position]].item())
            current_batch_size = adaptive_batch_size(current_len)
            batch_positions = length_sorted_order[position : position + current_batch_size]
            batch_max_len = int(real_lengths[batch_positions].max().item())

            batch = {
                "input_ids": input_ids[batch_positions, :batch_max_len],
                "attention_mask": attention_mask[batch_positions, :batch_max_len],
            }

            # Generate embeddings
            batch_embeddings = generate_embeddings_with_oom_retry(model, device, batch, EMBED_DIM)

            # Write to pre-allocated array at each item's original position
            position_indices = batch_positions.numpy()
            all_embeddings[position_indices] = batch_embeddings
            filled[position_indices] = True

            batch_num += 1
            logger.debug(
                "Embedded batch %d (%d items, batch_size=%d, trimmed to length %d)",
                batch_num, len(batch_positions), current_batch_size, batch_max_len,
            )
            progress_bar.update(len(batch_positions))
            position += len(batch_positions)

            if time.perf_counter() - last_checkpoint_time > CHECKPOINT_INTERVAL_SECONDS:
                _save_checkpoint(checkpoint_path, all_embeddings, filled, total_items)
                last_checkpoint_time = time.perf_counter()

    elapsed = time.perf_counter() - embed_start
    logger.info(
        "Generated %d embeddings in %.1fs (%.2f items/s)",
        total_items, elapsed, total_items / elapsed if elapsed else 0.0,
    )

    # Add embeddings to dataframe
    embeddings_list = all_embeddings.tolist()
    items_df_with_emb = item_df.with_columns(pl.Series("embedding", embeddings_list, dtype=pl.List(pl.Float32)))

    items_df_with_emb.write_parquet(output_path)
    logger.info("Saved embeddings dataset to %s", output_path)

    checkpoint_path.unlink(missing_ok=True)

    return items_df_with_emb


if __name__ == "__main__":
    embed_items()