"""Builds Alpaca-format instruction-tuning data from the trained semantic IDs.

Four task types:
  - sequential: predict the next item's semantic ID from a user's play history
  - grounding: map a semantic ID <-> item name/genres, both directions
  - similar: given an item, suggest another one real users also engaged with
    (ground truth from co-occurrence in user sequences, not just genre overlap)
  - asy (asymmetric item prediction, from LC-Rec -- arXiv 2311.09049): same
    (history, target) pairs as sequential, but the target is rendered as its
    name+genres text instead of its semantic ID. Reuses sequential's much
    larger, more-repeated example pool to reinforce the index<->language link
    grounding needs, instead of leaving that link isolated in grounding's own
    sparse examples.

Two further fixes, after a full two-stage fine-tune (codebook-grounded init,
full-sequence loss, bigger batch, more epochs) still mode-collapsed onto a
handful of fixed default answers instead of learning per-item associations:

1. Catalog restricted to the ~8.5k items that actually appear in a user
   sequence, not the full 93k-item catalog. Two problems this fixes at once:
   (a) grounding examples were 98.5% single-exposure (each item's name<->ID
   pair seen once per epoch, 3 times total across a 3-epoch run) -- nowhere
   near enough repetition to memorize ~90k arbitrary associations. Shrinking
   the catalog ~11x means the same data volume now gives ~11x more exposure
   per item. (b) it also matches the catalog scale of reference projects
   (LC-Rec's Instruments dataset: 9,922 items) instead of being ~9x larger.
   The RQ-VAE codebook itself is reused as-is (not retrained) -- 256 codes
   per level is enormously more room than 8.5k items need, so collisions
   don't increase, and everything already built against it (the codebook-
   grounded initialization) stays valid.

2. Floor/ceiling rebalancing (see `_rebalance_by_target`). Real usage data is
   popularity-skewed -- measured on the pre-filter data, similar_item's top
   10 targets (of 681 unique) accounted for 80% of all examples, and the top
   2 were the exact sid sequences the trained model kept defaulting to
   regardless of input. That's not a training bug: always guessing the
   popular answer really does minimize average loss on a distribution that
   skewed, so a model with weak input-conditioning learns to do exactly
   that. Capping any single target's example count (ceiling) removes that
   shortcut; flooring under-represented targets (grounding's core problem)
   gives rare items enough repetition to actually be learnable.

Train/val splitting happens by GROUP (the same target_key_fn used for
rebalancing), not by individual example. Oversampled examples are near-
duplicates of each other (same input/output, varied instruction phrasing)
-- splitting at the example level could put 9 of an item's 10 repeats in
train and 1 in val, making val trivially easy rather than genuinely held
out. Splitting whole groups keeps val honest.

Each example is {"instruction", "input", "output", "task"} — "task" is metadata
beyond the strict 3-key Alpaca schema, kept for traceability; drop it if your
fine-tuning framework requires the exact format.
"""

import json
import random
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Callable, List, Optional

import polars as pl

from config import RQVAEConfig
from logger import Logger

logger = Logger.get_logger(__name__)

SID_START = "<|sid_start|>"
SID_END = "<|sid_end|>"

SEQUENTIAL_INSTRUCTIONS = [
    "Given a user's game history, ordered from most to least played, predict the semantic ID of the next game they are likely to enjoy.",
    "Here is a list of games a player has spent time on, from most to least played. What game's semantic ID would you recommend next?",
    "Based on this player's play history (most engaged first), predict the semantic ID of a game they would likely enjoy next.",
]

ASY_INSTRUCTIONS = [
    "Given a user's game history, ordered from most to least played, predict the name of the next game they are likely to enjoy.",
    "Here is a list of games a player has spent time on, from most to least played. What game would you recommend next? Tell me its name.",
    "Based on this player's play history (most engaged first), predict the title of a game they would likely enjoy next.",
]

ID_TO_NAME_INSTRUCTIONS = [
    "What game does this semantic ID represent?",
    "Identify the game corresponding to this semantic ID.",
    "Which game is encoded by this semantic ID?",
]

NAME_TO_ID_INSTRUCTIONS = [
    "What is the semantic ID for this game?",
    "Give the semantic ID that represents this game.",
    "Encode this game as its semantic ID.",
]

SIMILAR_INSTRUCTIONS = [
    "A player enjoyed this game. Suggest another game they would likely also enjoy.",
    "Given a game a player liked, recommend a similar game.",
    "Players who played this game also played the following game. Name it by semantic ID.",
]


class AlpacaDatasetBuilder:
    def __init__(
        self,
        config: RQVAEConfig,
        semantic_ids_path: Path = None,
        catalog_path: Path = None,
        sequences_path: Path = None,
        output_dir: Path = None,
        max_history_items: int = 10,
        max_examples_per_user: int = 3,
        cooccurrence_window: int = 30,
        # Was 3. The catalog is now ~11x smaller (8.5k interacted items, not
        # 93k) so there's much less combinatorial space for co-occurrence to
        # spread across -- affordable to require less confidence per pair
        # while still boosting similar_item's raw volume before rebalancing
        # caps it (similar_item was the smallest task by far, ~11k examples).
        min_cooccurrence: int = 2,
        # Was 5, for the same reason as min_cooccurrence.
        max_similar_per_item: int = 10,
        exclude_long_tail_users: bool = True,
        restrict_to_interacted_items: bool = True,
        # Grounding/ASY: repeat each item's example(s) up to this many times
        # (varied instruction phrasing per repeat) so rare items get enough
        # exposure to actually be learnable -- was effectively 1 before.
        grounding_repeat_floor: int = 10,
        # sequential/ASY: floor and ceiling on how many times any single
        # target item can appear as the prediction target, applied to the
        # shared (history, target) pairs before rendering either task's
        # output. Placeholder values -- see build_finetune_dataset.py's
        # __main__ block / the accompanying distribution-inspection pass for
        # the data-driven numbers actually used.
        sequential_target_floor: int = 5,
        sequential_target_ceiling: int = 50,
        # similar_item: same idea, its own floor/ceiling since its raw
        # distribution is far more skewed (80% of examples in the top 10
        # targets, out of 681 unique, measured pre-filter) than sequential's.
        similar_target_floor: int = 3,
        similar_target_ceiling: int = 20,
        val_split: float = None,
        seed: int = 0,
    ):
        self.config = config
        self.semantic_ids_path = semantic_ids_path or config.data_dir / "output" / "semantic_ids.parquet"
        self.catalog_path = catalog_path or config.data_dir / "clean_game_catalog.parquet"
        self.sequences_path = sequences_path or config.data_dir / "clean_user_sequences.parquet"
        self.output_dir = Path(output_dir) if output_dir else config.data_dir / "output"

        self.max_history_items = max_history_items
        self.max_examples_per_user = max_examples_per_user
        self.cooccurrence_window = cooccurrence_window
        self.min_cooccurrence = min_cooccurrence
        self.max_similar_per_item = max_similar_per_item
        self.exclude_long_tail_users = exclude_long_tail_users
        self.restrict_to_interacted_items = restrict_to_interacted_items
        self.grounding_repeat_floor = grounding_repeat_floor
        self.sequential_target_floor = sequential_target_floor
        self.sequential_target_ceiling = sequential_target_ceiling
        self.similar_target_floor = similar_target_floor
        self.similar_target_ceiling = similar_target_ceiling
        self.val_split = val_split if val_split is not None else config.val_split

        self.rng = random.Random(seed)

        self.sequences_df: pl.DataFrame = None
        self.item_tokens: dict = {}  # id -> "<|sid_start|>...<|sid_end|>"
        self.item_name: dict = {}    # id -> Name
        self.item_desc: dict = {}    # id -> "Name — Genre, Genre" style description

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def get_special_tokens(self) -> List[str]:
        """Every token that must be added to the tokenizer before fine-tuning:
        one per (level, code) pair, plus the start/end markers. Independent
        of which items are actually used -- this is the full RQ-VAE code
        space, not a per-item enumeration."""
        n_levels = self.config.codebook_quantization_levels + 1  # +1 for the disambiguation digit
        tokens = [SID_START, SID_END]
        for level in range(n_levels):
            for code in range(self.config.codebook_size):
                tokens.append(f"<|sid_L{level}_{code}|>")
        return tokens

    def semantic_id_to_tokens(self, semantic_id: List[int]) -> str:
        levels = "".join(f"<|sid_L{level}_{code}|>" for level, code in enumerate(semantic_id))
        return f"{SID_START}{levels}{SID_END}"

    def load_data(self):
        logger.info("Loading semantic IDs from %s", self.semantic_ids_path)
        sid_df = pl.read_parquet(self.semantic_ids_path)

        logger.info("Loading catalog from %s", self.catalog_path)
        catalog_df = pl.read_parquet(self.catalog_path, columns=["id", "Name", "Genres"])

        logger.info("Loading user sequences from %s", self.sequences_path)
        self.sequences_df = pl.read_parquet(self.sequences_path)

        joined = sid_df.join(catalog_df, on="id", how="inner")
        logger.info("Joined %d items (%d semantic IDs, %d catalog rows)", len(joined), len(sid_df), len(catalog_df))

        if self.restrict_to_interacted_items:
            interacted_ids = set()
            for row in self.sequences_df.iter_rows(named=True):
                interacted_ids.update(row["item_sequence"])
            before = len(joined)
            joined = joined.filter(pl.col("id").is_in(interacted_ids))
            logger.info(
                "Restricted catalog to items appearing in a user sequence: %d -> %d items",
                before, len(joined),
            )

        for row in joined.iter_rows(named=True):
            item_id = row["id"]
            self.item_tokens[item_id] = self.semantic_id_to_tokens(row["semantic_ids"])
            self.item_name[item_id] = row["Name"]
            genres = row["Genres"].replace(",", ", ") if row["Genres"] else None
            self.item_desc[item_id] = f"{row['Name']} — {genres}" if genres else row["Name"]

        logger.info("Indexed %d items", len(self.item_tokens))

    # ------------------------------------------------------------------
    # Rebalancing
    # ------------------------------------------------------------------

    def _rebalance_by_target(
        self,
        examples: List[dict],
        target_key_fn: Callable[[dict], object],
        floor: int,
        ceiling: int,
        instruction_pool: Optional[List[str]] = None,
    ) -> List[dict]:
        """Groups examples by target_key_fn(example), then subsamples any
        group above `ceiling` and oversamples (repeats, re-rolling the
        instruction from `instruction_pool` for phrasing variety) any group
        below `floor`. Every group's final count lands in [floor, ceiling]
        (or stays as-is if already within range)."""
        groups = defaultdict(list)
        for ex in examples:
            groups[target_key_fn(ex)].append(ex)

        rebalanced = []
        for _, group in groups.items():
            if len(group) > ceiling:
                rebalanced.extend(self.rng.sample(group, ceiling))
            elif len(group) < floor:
                rebalanced.extend(group)
                for _ in range(floor - len(group)):
                    clone = dict(self.rng.choice(group))
                    if instruction_pool:
                        clone["instruction"] = self.rng.choice(instruction_pool)
                    rebalanced.append(clone)
            else:
                rebalanced.extend(group)

        self.rng.shuffle(rebalanced)
        return rebalanced

    # ------------------------------------------------------------------
    # Task builders
    # ------------------------------------------------------------------

    def _build_history_target_pairs(self) -> List[tuple]:
        """Shared (history_item_ids, target_item_id) pairs consumed by both
        sequential and ASY -- same sampling, different output representation
        of the same target."""
        pairs = []
        skipped_users = 0

        for row in self.sequences_df.iter_rows(named=True):
            if self.exclude_long_tail_users and row["is_long_tail_user"]:
                skipped_users += 1
                continue

            sequence = [i for i in row["item_sequence"] if i in self.item_tokens]
            if len(sequence) < 2:
                continue

            positions = list(range(1, len(sequence)))
            if len(positions) > self.max_examples_per_user:
                positions = sorted(self.rng.sample(positions, self.max_examples_per_user))

            for pos in positions:
                history = sequence[max(0, pos - self.max_history_items):pos]
                target = sequence[pos]
                pairs.append((history, target))

        logger.info("Built %d history/target pairs (skipped %d long-tail users)", len(pairs), skipped_users)
        return pairs

    def build_sequential_and_asy_examples(self, pairs: List[tuple]) -> tuple:
        sequential, asy = [], []
        for history, target in pairs:
            history_tokens = " ".join(self.item_tokens[i] for i in history)
            sequential.append({
                "instruction": self.rng.choice(SEQUENTIAL_INSTRUCTIONS),
                "input": history_tokens,
                "output": self.item_tokens[target],
                "task": "sequential",
                "_target": target,
            })
            asy.append({
                "instruction": self.rng.choice(ASY_INSTRUCTIONS),
                "input": history_tokens,
                "output": self.item_desc[target],
                "task": "asy",
                "_target": target,
            })
        return sequential, asy

    def build_grounding_examples(self) -> tuple:
        id2name, name2id = [], []
        for item_id, tokens in self.item_tokens.items():
            id2name.append({
                "instruction": self.rng.choice(ID_TO_NAME_INSTRUCTIONS),
                "input": tokens,
                "output": self.item_desc[item_id],
                "task": "grounding_id2name",
                "_target": item_id,
            })
            name2id.append({
                "instruction": self.rng.choice(NAME_TO_ID_INSTRUCTIONS),
                "input": self.item_name[item_id],
                "output": tokens,
                "task": "grounding_name2id",
                "_target": item_id,
            })
        logger.info("Built %d raw grounding examples (%d items x 2 directions)", len(id2name) + len(name2id), len(self.item_tokens))
        return id2name, name2id

    def build_similar_examples(self) -> List[dict]:
        logger.info(
            "Computing item co-occurrence (window=%d, min_count=%d)...",
            self.cooccurrence_window, self.min_cooccurrence,
        )
        cooccurrence = Counter()
        skipped_users = 0

        for row in self.sequences_df.iter_rows(named=True):
            if self.exclude_long_tail_users and row["is_long_tail_user"]:
                skipped_users += 1
                continue

            items = [i for i in row["item_sequence"][: self.cooccurrence_window] if i in self.item_tokens]
            if len(items) < 2:
                continue
            for a, b in combinations(sorted(set(items)), 2):
                cooccurrence[(a, b)] += 1

        logger.info("Found %d co-occurring item pairs (skipped %d long-tail users)", len(cooccurrence), skipped_users)

        # Top co-occurring partner(s) per item, symmetric.
        partners: dict = {}
        for (a, b), count in cooccurrence.items():
            if count < self.min_cooccurrence:
                continue
            partners.setdefault(a, []).append((b, count))
            partners.setdefault(b, []).append((a, count))

        examples = []
        for item_id, candidates in partners.items():
            candidates.sort(key=lambda x: x[1], reverse=True)
            for partner_id, _count in candidates[: self.max_similar_per_item]:
                examples.append({
                    "instruction": self.rng.choice(SIMILAR_INSTRUCTIONS),
                    "input": self.item_tokens[item_id],
                    "output": self.item_tokens[partner_id],
                    "task": "similar_item",
                    "_target": partner_id,
                })

        logger.info(
            "Built %d raw similar-item examples from %d items with qualifying co-occurring partners",
            len(examples), len(partners),
        )
        return examples

    # ------------------------------------------------------------------
    # Assembly
    # ------------------------------------------------------------------

    def train_val_split_by_group(self, examples: List[dict]) -> tuple:
        """Splits by the same group (`_target`) used for rebalancing, not by
        individual example -- oversampled examples are near-duplicates of
        each other, so splitting at the example level could leak most of a
        group into train and leave val trivially easy."""
        groups = defaultdict(list)
        for ex in examples:
            groups[ex["_target"]].append(ex)

        group_keys = list(groups.keys())
        self.rng.shuffle(group_keys)
        n_val_groups = max(1, int(len(group_keys) * self.val_split))
        val_keys = set(group_keys[:n_val_groups])

        train, val = [], []
        for key, group in groups.items():
            (val if key in val_keys else train).extend(group)
        return train, val

    def build_all(self) -> dict:
        self.load_data()

        pairs = self._build_history_target_pairs()
        pairs = self._rebalance_pairs_by_target(pairs, self.sequential_target_floor, self.sequential_target_ceiling)
        sequential, asy = self.build_sequential_and_asy_examples(pairs)

        id2name, name2id = self.build_grounding_examples()
        id2name = self._rebalance_by_target(
            id2name, lambda ex: ex["_target"], self.grounding_repeat_floor, len(id2name), ID_TO_NAME_INSTRUCTIONS,
        )
        name2id = self._rebalance_by_target(
            name2id, lambda ex: ex["_target"], self.grounding_repeat_floor, len(name2id), NAME_TO_ID_INSTRUCTIONS,
        )

        similar = self.build_similar_examples()
        similar = self._rebalance_by_target(
            similar, lambda ex: ex["_target"], self.similar_target_floor, self.similar_target_ceiling, SIMILAR_INSTRUCTIONS,
        )

        tasks = {
            "sequential": sequential, "asy": asy,
            "grounding_id2name": id2name, "grounding_name2id": name2id,
            "similar_item": similar,
        }

        train_all, val_all = [], []
        for name, examples in tasks.items():
            train, val = self.train_val_split_by_group(examples)
            train_all.extend(train)
            val_all.extend(val)
            logger.info("%s: %d train, %d val", name, len(train), len(val))

        self.rng.shuffle(train_all)
        self.rng.shuffle(val_all)

        # "_target" is an internal grouping key, not part of the Alpaca schema.
        for ex in train_all + val_all:
            del ex["_target"]

        self.output_dir.mkdir(parents=True, exist_ok=True)
        train_path = self.output_dir / "sft_train.jsonl"
        val_path = self.output_dir / "sft_val.jsonl"
        self._write_jsonl(train_all, train_path)
        self._write_jsonl(val_all, val_path)

        special_tokens = self.get_special_tokens()
        special_tokens_path = self.output_dir / "sft_special_tokens.json"
        with open(special_tokens_path, "w", encoding="utf-8") as f:
            json.dump(special_tokens, f)
        logger.info("Saved %d special tokens to %s", len(special_tokens), special_tokens_path)

        logger.info(
            "Total: %d train, %d val examples -> %s, %s",
            len(train_all), len(val_all), train_path, val_path,
        )
        return {"train": train_all, "val": val_all}

    def _rebalance_pairs_by_target(self, pairs: List[tuple], floor: int, ceiling: int) -> List[tuple]:
        """Same floor/ceiling idea as _rebalance_by_target, applied to the
        raw (history, target) pairs before either sequential or ASY renders
        them -- keeps both tasks' target distributions identical rather than
        rebalancing them independently."""
        groups = defaultdict(list)
        for pair in pairs:
            groups[pair[1]].append(pair)

        rebalanced = []
        for _, group in groups.items():
            if len(group) > ceiling:
                rebalanced.extend(self.rng.sample(group, ceiling))
            elif len(group) < floor:
                rebalanced.extend(group)
                for _ in range(floor - len(group)):
                    rebalanced.append(self.rng.choice(group))
            else:
                rebalanced.extend(group)

        self.rng.shuffle(rebalanced)
        return rebalanced

    @staticmethod
    def _write_jsonl(examples: List[dict], path: Path):
        with open(path, "w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        logger.info("Wrote %d examples to %s", len(examples), path)


if __name__ == "__main__":
    config = RQVAEConfig()
    builder = AlpacaDatasetBuilder(config)
    builder.build_all()
