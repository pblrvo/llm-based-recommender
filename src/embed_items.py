import os
import time
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel

from logger import Logger

DATA_DIR = Path("data")

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
BATCH_SIZE = 64
EMBED_DIM = 1024

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


class TokenizedDataset(Dataset):
    def __init__(self, input_ids: Tensor, attention_mask: Tensor):
        self.input_ids = input_ids
        self.attention_mask = attention_mask

    def __len__(self):
        return self.input_ids.size(0)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
        }
    

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
    logger.info("Pre-tokenized data validated: %d items, sequence length %d", total_items, input_ids.shape[1])

    # Create dataset and dataloader
    dataset = TokenizedDataset(input_ids, attention_mask)

    num_workers = min(4, total_items)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        prefetch_factor=2 if num_workers else None,
        persistent_workers=num_workers > 0,
    )
    logger.info(
        "Generating embeddings: batch_size=%d, num_workers=%d, embed_dim=%d",
        BATCH_SIZE, num_workers, EMBED_DIM,
    )

    #Pre-allocate output array
    all_embeddings = np.zeros((total_items, EMBED_DIM), dtype=np.float32)

    current_idx = 0
    embed_start = time.perf_counter()
    with tqdm(total=total_items, desc="Generating Embeddings") as progress_bar:
        for batch_idx, batch in enumerate(dataloader):
            # Get batch size
            batch_size = batch["input_ids"].size(0)

            # Generate embeddings
            batch_embeddings = generate_embeddings(model, device, batch, EMBED_DIM)

            # Write to pre-allocated array
            all_embeddings[current_idx : current_idx + batch_size] = batch_embeddings
            current_idx += batch_size

            logger.debug("Embedded batch %d (%d items, %d/%d total)", batch_idx + 1, batch_size, current_idx, total_items)
            progress_bar.update(batch_size)

    elapsed = time.perf_counter() - embed_start
    logger.info(
        "Generated %d embeddings in %.1fs (%.2f items/s)",
        total_items, elapsed, total_items / elapsed if elapsed else 0.0,
    )

    # Add embeddings to dataframe
    embeddings_list = all_embeddings.tolist()
    items_df_with_emb = item_df.with_columns(pl.Series("embedding", embeddings_list, dtype=pl.List(pl.Float32)))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    items_df_with_emb.write_parquet(output_path)
    logger.info("Saved embeddings dataset to %s", output_path)

    return items_df_with_emb


if __name__ == "__main__":
    embed_items()