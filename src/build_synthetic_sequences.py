"""Generates synthetic user sequences to top up under-exposed items for the
sequential task: a temperature-weighted random walk over a k-NN graph built
from item embeddings, targeted at items below `target_floor`. Output is
schema-compatible with clean_user_sequences.parquet, tagged with an extra
is_synthetic column; main() writes both the synthetic-only file and a
combined (real + synthetic) file for build_finetune_dataset.py.
"""

import argparse
import collections

import numpy as np
import polars as pl
from sklearn.neighbors import NearestNeighbors

from config import RQVAEConfig
from logger import Logger

logger = Logger.get_logger(__name__)


def played_sequence(row: dict, interacted_ids: set) -> list:
    """Item IDs from `row`, restricted to played (nonzero playtime) catalog items."""
    return [
        item_id
        for item_id, playtime in zip(row["item_sequence"], row["playtime_sequence"])
        if playtime > 0 and item_id in interacted_ids
    ]


def load_real_frequencies(sequences_df: pl.DataFrame, interacted_ids: set) -> collections.Counter:
    """Count how often each item appears as a target position (index >= 1) in a real sequence."""
    counts = collections.Counter()
    for row in sequences_df.iter_rows(named=True):
        if row["is_long_tail_user"]:
            continue
        counts.update(played_sequence(row, interacted_ids)[1:])
    return counts


def build_knn(embeddings: np.ndarray, k: int) -> NearestNeighbors:
    """Fit a k-NN index over the item embeddings using cosine distance."""
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine")
    nn.fit(embeddings)
    return nn


def walk_to_target(
    target_idx: int,
    length: int,
    neighbor_idx: np.ndarray,
    neighbor_sim: np.ndarray,
    temperature: float,
    rng: np.random.Generator,
) -> list:
    """Build a length-`length` walk ending at target_idx by stepping through neighbors backward."""
    walk = [target_idx]
    current = target_idx
    for _ in range(length - 1):
        neighbors = neighbor_idx[current][1:]  # drop self (nearest neighbor is always self)
        sims = neighbor_sim[current][1:]
        weights = np.exp(sims / temperature)
        weights /= weights.sum()
        current = rng.choice(neighbors, p=weights)
        walk.append(current)
    walk.reverse()
    return walk


def main():
    """CLI entry point: build the k-NN graph, walk to under-exposed items, and write the synthetic + combined files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-floor", type=int, default=5, help="Match sequential_target_floor.")
    parser.add_argument("--k", type=int, default=15, help="Neighbors per item in the k-NN graph.")
    parser.add_argument("--temperature", type=float, default=0.1, help="Softmax temperature over cosine sim.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config = RQVAEConfig()
    sequences_path = config.data_dir / "clean_user_sequences.parquet"
    embeddings_path = config.data_dir / "output" / "games_with_embeddings.parquet"
    output_path = args.output or (config.data_dir / "synthetic_user_sequences.parquet")

    rng = np.random.default_rng(args.seed)

    logger.info("Loading real sequences from %s", sequences_path)
    sequences_df = pl.read_parquet(sequences_path)

    logger.info("Loading item embeddings from %s", embeddings_path)
    items_df = pl.read_parquet(embeddings_path, columns=["id", "embedding"])

    interacted_ids = set()
    for row in sequences_df.iter_rows(named=True):
        interacted_ids.update(row["item_sequence"])
    items_df = items_df.filter(pl.col("id").is_in(interacted_ids))
    logger.info("Restricted embedding index to %d interacted items", len(items_df))

    item_ids = items_df["id"].to_list()
    embeddings = np.stack(items_df["embedding"].to_list()).astype(np.float32)
    id_to_idx = {item_id: idx for idx, item_id in enumerate(item_ids)}

    lengths = [
        len(played_sequence(row, interacted_ids))
        for row in sequences_df.iter_rows(named=True)
        if not row["is_long_tail_user"]
    ]
    lengths = [l for l in lengths if l >= 2]
    logger.info("Real played-sequence lengths: median=%d, p10=%d, p90=%d (n=%d)",
                int(np.median(lengths)), int(np.percentile(lengths, 10)),
                int(np.percentile(lengths, 90)), len(lengths))

    real_freq = load_real_frequencies(sequences_df, interacted_ids)
    under_exposed = {
        item_id: args.target_floor - real_freq.get(item_id, 0)
        for item_id in item_ids
        if real_freq.get(item_id, 0) < args.target_floor
    }
    total_needed = sum(under_exposed.values())
    logger.info("%d/%d items under the floor of %d, %d synthetic sequences needed",
                len(under_exposed), len(item_ids), args.target_floor, total_needed)

    nn = build_knn(embeddings, args.k)
    neighbor_sim, neighbor_idx = nn.kneighbors(embeddings)
    neighbor_sim = 1.0 - neighbor_sim  # cosine distance -> similarity

    synthetic_rows = []
    for item_id, need in under_exposed.items():
        target_idx = id_to_idx[item_id]
        for _ in range(need):
            length = int(rng.choice(lengths))
            walk_idx = walk_to_target(target_idx, length, neighbor_idx, neighbor_sim, args.temperature, rng)
            walk_ids = [item_ids[i] for i in walk_idx]
            synthetic_rows.append({
                "item_sequence": walk_ids,
                "playtime_sequence": [1] * len(walk_ids),  # placeholder positive playtime
                "is_long_tail_user": False,
            })

    remaining_cols = set(sequences_df.columns) - {"item_sequence", "playtime_sequence", "is_long_tail_user"}
    for row in synthetic_rows:
        for col in remaining_cols:
            row[col] = None

    synthetic_df = pl.DataFrame(synthetic_rows, schema=sequences_df.schema)
    synthetic_df = synthetic_df.with_columns(pl.lit(True).alias("is_synthetic"))
    synthetic_df.write_parquet(output_path)
    logger.info("Wrote %d synthetic sequences to %s", len(synthetic_df), output_path)

    real_tagged = sequences_df.with_columns(pl.lit(False).alias("is_synthetic"))
    combined_df = pl.concat([real_tagged, synthetic_df.select(real_tagged.columns)])
    combined_path = output_path.parent / "combined_user_sequences.parquet"
    combined_df.write_parquet(combined_path)
    logger.info(
        "Wrote %d combined sequences (%d real + %d synthetic) to %s",
        len(combined_df), len(real_tagged), len(synthetic_df), combined_path,
    )


if __name__ == "__main__":
    main()
