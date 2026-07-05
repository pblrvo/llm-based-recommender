import os
from pathlib import Path
from typing import List

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm
import polars as pl

DATA_DIR = Path("data")
MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
BATCH_SIZE = 32
MAX_LENGTH = 2000


def get_instruction(task, text):

    return f"Instruct: {task}\nQuery: {text}"

def tokenize_and_save_embeddings(item_texts: List[str], tokenizer, max_length: int, batch_size: int, output_path: Path, total_items: int):
    all_input_ids = []
    all_attention_masks = []

    task = (
        "Given a video game description, generate a semantic embedding that captures the essence of the game, including its genre, tags, key features and characteristics."
    )

    for i in tqdm(range(0, total_items, batch_size), desc="Tokenizing"):
        batch_texts = item_texts[i : i + batch_size]
        instructions = [get_instruction(task, text) for text in batch_texts]

        # Tokenize text
        encoded = tokenizer(
            instructions,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        all_input_ids.append(encoded["input_ids"].numpy())
        all_attention_masks.append(encoded["attention_mask"].numpy())

    # Concatenate all batches
    input_ids = np.vstack(all_input_ids)
    attention_mask = np.vstack(all_attention_masks)

    np.savez_compressed(
        output_path,
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        n_items=total_items
    )

    return input_ids.shape


def tokenize_items(input_path: Path = None, output_path: Path = None, limit: int = None, max_length: int = None):
    input_path = input_path or DATA_DIR / "clean_game_catalog.parquet"
    output_path = output_path or DATA_DIR / "tokenized_game_catalog.npz"
    max_length = max_length or MAX_LENGTH

    #Load data
    item_df = pl.read_parquet(input_path)
    pl.Config.set_fmt_str_lengths(2000)

    if limit:
        item_df = item_df.head(limit)

    total_items = len(item_df)

    item_text_for_emb = item_df["text_for_embedding"].to_list()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    return tokenize_and_save_embeddings(item_text_for_emb, tokenizer, max_length, BATCH_SIZE, output_path, total_items)


if __name__ == "__main__":
    tokenize_items()