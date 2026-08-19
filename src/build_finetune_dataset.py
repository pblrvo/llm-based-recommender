"""Builds Alpaca-format instruction-tuning data from the trained semantic IDs.

Eight task types:
  - sequential: predict the next item's semantic ID from a user's play history
  - grounding: map a semantic ID <-> item name/genres, both directions
  - similar / nl_similar_item: given an item (or a natural-language reference
    to it), suggest another one real users also engaged with (ground truth
    from PMI-ranked co-occurrence in user sequences)
  - asy (asymmetric item prediction, from LC-Rec -- arXiv 2311.09049): same
    (history, target) pairs as sequential, target rendered as name+genres
  - nl_preference: open-ended natural-language preference queries -> a real
    matching item's semantic ID; many items validly satisfy one query, so
    several different real targets are shown per query type, and each
    example carries an extra "criteria" field checked for genre/category
    consistency rather than exact-match recall
  - relatedness: given two semantic IDs, predict whether they share a
    level-0 codebook code (see build_relatedness_examples.py)

Catalog is restricted to the ~8.5k items that actually appear in a user
sequence (not the full ~93k), and floor/ceiling rebalancing
(`_rebalance_by_target`) plus a cross-task exposure cap
(`_cap_total_exposure_across_tasks`) keep any single item from dominating
the recommendation-shaped tasks' training signal.

Train/val splitting happens by target GROUP for the relational tasks
(sequential/asy/similar_item/nl_similar_item/nl_preference), so oversampled
near-duplicates don't leak across the split. grounding_name2id/
grounding_id2name split WITHIN each item's group instead
(train_val_split_within_group), since grounding is closer to an exhaustive
lookup table than a generalization task.

Each example is {"instruction", "input", "output", "task"}; nl_preference
examples carry one further "criteria" field. Both "task" and "criteria" are
eval-time metadata beyond the strict 3-key Alpaca schema.
"""

import json
import math
import random
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Callable, List, Optional

import polars as pl

from build_relatedness_examples import build_relatedness_examples
from config import RQVAEConfig
from logger import Logger

logger = Logger.get_logger(__name__)

SID_START = "<|sid_start|>"
SID_END = "<|sid_end|>"

SEQUENTIAL_INSTRUCTIONS = [
    "Given a user's game history, ordered from most to least played, predict the semantic ID of the next game they are likely to enjoy.",
    "Here is a list of games a player has spent time on, from most to least played. What game's semantic ID would you recommend next?",
    "Based on this player's play history (most engaged first), predict the semantic ID of a game they would likely enjoy next.",
    "This player's games are listed from most to least played. Predict the semantic ID of what they'll play next.",
    "Looking at this engagement-ordered play history, what semantic ID would you recommend next?",
    "Given the games below (most played first), output the semantic ID of the next game for this player.",
    "A player's history is shown ranked by engagement. Predict their next game's semantic ID.",
    "Using this player's most-to-least-played game list, forecast the semantic ID of their next pick.",
    "Here's what this player has played, ranked by time invested. What's the semantic ID of a game they'd play next?",
    "From this ranked play history, infer the semantic ID of the next game this player would enjoy.",
    "This is a player's game history in order of engagement. Give the semantic ID of your next recommendation.",
    "Predict the next game's semantic ID for a player with this play history (most engaged first).",
]

ASY_INSTRUCTIONS = [
    "Given a user's game history, ordered from most to least played, predict the name of the next game they are likely to enjoy.",
    "Here is a list of games a player has spent time on, from most to least played. What game would you recommend next? Tell me its name.",
    "Based on this player's play history (most engaged first), predict the title of a game they would likely enjoy next.",
    "This player's games are listed from most to least played. Name the next game they'll likely play.",
    "Looking at this engagement-ordered play history, which game would you recommend next? Give its name.",
    "Given the games below (most played first), name the next game for this player.",
    "A player's history is shown ranked by engagement. Predict the name of their next game.",
    "Using this player's most-to-least-played game list, name your next recommendation.",
    "Here's what this player has played, ranked by time invested. What game should they play next?",
    "From this ranked play history, name the next game this player would enjoy.",
    "This is a player's game history in order of engagement. Give the title of your next recommendation.",
    "Predict the name of the next game for a player with this play history (most engaged first).",
]

ID_TO_NAME_INSTRUCTIONS = [
    "What game does this semantic ID represent?",
    "Identify the game corresponding to this semantic ID.",
    "Which game is encoded by this semantic ID?",
    "Decode this semantic ID into the game it stands for.",
    "Tell me the name of the game behind this semantic ID.",
    "This semantic ID maps to a specific game. Which one?",
    "Translate this semantic ID back into a game name.",
    "What is this semantic ID's corresponding game?",
    "Given this semantic ID, name the game it identifies.",
    "Look up the game associated with this semantic ID.",
]

NAME_TO_ID_INSTRUCTIONS = [
    "What is the semantic ID for this game?",
    "Give the semantic ID that represents this game.",
    "Encode this game as its semantic ID.",
    "Convert this game into its semantic ID.",
    "Look up the semantic ID for this game.",
    "This game has a corresponding semantic ID. What is it?",
    "Translate this game's name into its semantic ID.",
    "Identify the semantic ID that encodes this game.",
    "What semantic ID does this game map to?",
    "Provide the semantic ID for the following game.",
]

SIMILAR_INSTRUCTIONS = [
    "A player enjoyed this game. Suggest another game they would likely also enjoy.",
    "Given a game a player liked, recommend a similar game.",
    "Players who played this game also played the following game. Name it by semantic ID.",
    "This game was a hit with a player. What similar game's semantic ID would you suggest?",
    "Recommend, by semantic ID, a game similar to the one below.",
    "A player liked this game a lot. Give the semantic ID of something similar.",
    "Based on this game, suggest another one real players also enjoyed. Give its semantic ID.",
    "What game's semantic ID would you pair with this one for a similar player?",
    "Given this game a player enjoyed, what's a good next pick? Answer with its semantic ID.",
    "Suggest, by semantic ID, a game that fans of this one also tend to like.",
]

NL_QUERY_INSTRUCTIONS = [
    "A player describes what kind of game they want to play. Recommend a matching game by its semantic ID.",
    "Based on this player's request, suggest a game that fits by giving its semantic ID.",
    "Given the following game preference, recommend a matching game's semantic ID.",
    "A player states what they're in the mood for. Give the semantic ID of a game that matches.",
    "Read this player's preference and respond with a matching game's semantic ID.",
    "This player wants a specific kind of game. Recommend one by semantic ID.",
    "Match this game request to a real catalog item. Answer with its semantic ID.",
    "Given what this player is looking for, suggest a fitting game's semantic ID.",
    "A player's game preference is described below. Give the semantic ID of a suitable match.",
]

# {article} is "a"/"an", computed by AlpacaDatasetBuilder._indefinite_article.
GENRE_QUERY_TEMPLATES = [
    "I want to play {article} {genre} game.",
    "Recommend me {article} {genre} game.",
    "I'm looking for something in the {genre} genre.",
    "Suggest a good {genre} game to play.",
    "Can you recommend {article} {genre} game?",
    "Got any {genre} games worth playing?",
    "I'm in the mood for {article} {genre} game.",
    "What's a solid {genre} game I could try?",
    "I feel like playing {article} {genre} game right now.",
    "Any suggestions for {article} {genre} game?",
    "Point me to {article} {genre} game.",
    "Show me a {genre} game I might like.",
]

GENRE_COMBO_QUERY_TEMPLATES = [
    "I want to play {article1} {genre1} {genre2} game.",
    "Recommend a game that's both {genre1} and {genre2}.",
    "Looking for {article1} {genre1}/{genre2} game recommendation.",
    "Got any games that mix {genre1} and {genre2}?",
    "I want something that's {genre1} and {genre2} at the same time.",
    "Suggest {article1} {genre1}-{genre2} game.",
    "What's a good {genre1} and {genre2} crossover game?",
]

CATEGORY_QUERY_TEMPLATES = [
    "I want to play {article} {genre} game with {category}.",
    "Recommend {article} {genre} game that supports {category}.",
    "Looking for a {category} {genre} game.",
    "I want {article} {genre} game I can play {category}.",
    "Suggest {article} {genre} game with {category} support.",
    "Any {category} {genre} games you'd recommend?",
    "I'm after {article} {genre} game with {category}.",
    "Got a {category} {genre} game in mind?",
]

# Maps the raw catalog value to how it reads naturally inside CATEGORY_QUERY_TEMPLATES.
RELEVANT_CATEGORIES = {
    "Multi-player": "multiplayer",
    "Co-op": "co-op",
    "PvP": "PvP",
    "Single-player": "singleplayer",
}

# Below this many matching items in the catalog, a genre/combo is too sparse for several distinct answers.
MIN_GENRE_ITEM_COUNT = 100
MIN_COMBO_ITEM_COUNT = 15

MAX_BLURB_WORDS = 30  # catalog descriptions run to hundreds of words, far past the sequence budget

NL_SIMILAR_INSTRUCTIONS = [
    "A player enjoyed a game and describes it by name. Recommend a similar game by its semantic ID.",
    "Given the name of a game a player liked, suggest a similar game's semantic ID.",
    "A player names a game they liked. Give the semantic ID of something similar.",
    "This player enjoyed the named game. Recommend a similar one by semantic ID.",
    "Based on the game named below, suggest a similar game's semantic ID.",
    "A player liked this game (given by name). What similar game's semantic ID fits?",
]

SIMILAR_NL_TEMPLATES = [
    "Recommend me a game similar to {item_name}.",
    "I want a game like {item_name}.",
    "What's a good game similar to {item_name}?",
    "Suggest something similar to {item_name}.",
    "I enjoyed {item_name}. What should I play next?",
    "Give me something in the same vein as {item_name}.",
    "If I liked {item_name}, what else would I enjoy?",
    "Got anything like {item_name}?",
    "{item_name} was great. What's similar?",
    "I'm looking for a game similar to {item_name}.",
    "What would you recommend to someone who loved {item_name}?",
]


class AlpacaDatasetBuilder:
    """Builds Alpaca-format SFT examples from semantic IDs, the catalog, and user sequences."""

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
        min_cooccurrence: int = 2,
        max_similar_per_item: int = 10,
        exclude_long_tail_users: bool = True,
        restrict_to_interacted_items: bool = True,
        grounding_repeat_floor: int = 10,  # repeat each item's grounding example(s) up to this many times
        sequential_target_floor: int = 5,
        sequential_target_ceiling: int = 50,
        similar_target_floor: int = 3,
        similar_target_ceiling: int = 20,
        nl_examples_per_genre: int = 150,  # distinct real (query, target) pairs shown per genre, not a floor/ceiling
        nl_examples_per_combo: int = 40,
        relatedness_examples_per_item: int = 3,  # positive+negative pairs generated per item, not a floor/ceiling
        # Cap on an item's TOTAL appearance as a target, summed across every recommendation-shaped
        # task (see _cap_total_exposure_across_tasks). Not applied to grounding, which stays uniform.
        max_total_recommendation_exposure: int = 40,
        val_split: float = None,
        seed: int = 0,
    ):
        """Configure paths, hyperparameters, and the seeded RNG."""
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
        self.nl_examples_per_genre = nl_examples_per_genre
        self.nl_examples_per_combo = nl_examples_per_combo
        self.relatedness_examples_per_item = relatedness_examples_per_item
        self.max_total_recommendation_exposure = max_total_recommendation_exposure
        self.sequential_target_floor = sequential_target_floor
        self.sequential_target_ceiling = sequential_target_ceiling
        self.similar_target_floor = similar_target_floor
        self.similar_target_ceiling = similar_target_ceiling
        self.val_split = val_split if val_split is not None else config.val_split

        self.rng = random.Random(seed)

        self.sequences_df: pl.DataFrame = None
        self.item_tokens: dict = {}      # id -> "<|sid_start|>...<|sid_end|>"
        self.item_name: dict = {}        # id -> Name
        self.item_desc: dict = {}        # id -> "Name — Genre, Genre" style description
        self.item_genres: dict = {}      # id -> {genre, ...}
        self.item_categories: dict = {}  # id -> {category, ...}
        self.item_blurb: dict = {}       # id -> short snippet of "About the game"
        self.item_codes: dict = {}       # id -> tuple(level codes), e.g. (89, 210, 246, 0)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def get_special_tokens(self) -> List[str]:
        """Return every token that must be added to the tokenizer before fine-tuning: one per (level, code) pair plus start/end markers."""
        n_levels = self.config.codebook_quantization_levels + 1  # +1 for the disambiguation digit
        tokens = [SID_START, SID_END]
        for level in range(n_levels):
            for code in range(self.config.codebook_size):
                tokens.append(f"<|sid_L{level}_{code}|>")
        return tokens

    def semantic_id_to_tokens(self, semantic_id: List[int]) -> str:
        """Render a semantic ID as its full sid_start ... sid_end token string."""
        levels = "".join(f"<|sid_L{level}_{code}|>" for level, code in enumerate(semantic_id))
        return f"{SID_START}{levels}{SID_END}"

    def load_data(self):
        """Load semantic IDs, the catalog, and user sequences into in-memory dicts."""
        logger.info("Loading semantic IDs from %s", self.semantic_ids_path)
        sid_df = pl.read_parquet(self.semantic_ids_path)

        logger.info("Loading catalog from %s", self.catalog_path)
        catalog_df = pl.read_parquet(self.catalog_path, columns=["id", "Name", "Genres", "Categories", "About the game"])

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
            self.item_codes[item_id] = tuple(row["semantic_ids"])
            self.item_name[item_id] = row["Name"]
            genres = row["Genres"].replace(",", ", ") if row["Genres"] else None
            self.item_desc[item_id] = f"{row['Name']} — {genres}" if genres else row["Name"]
            self.item_genres[item_id] = {g.strip() for g in row["Genres"].split(",")} if row["Genres"] else set()
            self.item_categories[item_id] = {c.strip() for c in row["Categories"].split(",")} if row["Categories"] else set()
            self.item_blurb[item_id] = self._truncate_blurb(row["About the game"])

        logger.info("Indexed %d items", len(self.item_tokens))

    def _played_sequence(self, row: dict) -> List:
        """Item IDs from `row`, restricted to played items (nonzero playtime) known to the catalog."""
        return [
            item_id
            for item_id, playtime in zip(row["item_sequence"], row["playtime_sequence"])
            if playtime > 0 and item_id in self.item_tokens
        ]

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
        """Group examples by `target_key_fn`, then subsample over `ceiling` and oversample below `floor`.

        Oversampled rows are clones with a freshly re-rolled instruction from
        `instruction_pool`. Every group's final count lands in [floor, ceiling].
        """
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

    def _cap_total_exposure_across_tasks(self, task_examples: dict, max_total: int) -> dict:
        """Cap each item's TOTAL appearance as a target, pooled across every task in `task_examples`.

        Pools every (task, example) pair by `_target`; if an item's combined
        count exceeds `max_total`, randomly keeps `max_total` of them
        (irrespective of which task) and drops the rest.
        """
        pooled = defaultdict(list)
        for task_name, examples in task_examples.items():
            for ex in examples:
                pooled[ex["_target"]].append((task_name, ex))

        capped = {task_name: [] for task_name in task_examples}
        for _, items in pooled.items():
            if len(items) > max_total:
                items = self.rng.sample(items, max_total)
            for task_name, ex in items:
                capped[task_name].append(ex)

        for task_name in capped:
            self.rng.shuffle(capped[task_name])
        return capped

    # ------------------------------------------------------------------
    # Task builders
    # ------------------------------------------------------------------

    def _build_history_target_pairs(self) -> List[tuple]:
        """Build shared (history_item_ids, target_item_id, is_synthetic) pairs for sequential + asy."""
        pairs = []
        skipped_users = 0

        for row in self.sequences_df.iter_rows(named=True):
            if self.exclude_long_tail_users and row["is_long_tail_user"]:
                skipped_users += 1
                continue

            sequence = self._played_sequence(row)
            if len(sequence) < 2:
                continue

            is_synthetic = row.get("is_synthetic", False)
            positions = list(range(1, len(sequence)))
            if len(positions) > self.max_examples_per_user:
                positions = sorted(self.rng.sample(positions, self.max_examples_per_user))

            for pos in positions:
                history = sequence[max(0, pos - self.max_history_items):pos]
                target = sequence[pos]
                pairs.append((history, target, is_synthetic))

        logger.info("Built %d history/target pairs (skipped %d long-tail users)", len(pairs), skipped_users)
        return pairs

    def build_sequential_and_asy_examples(self, pairs: List[tuple]) -> tuple:
        """Render shared history->target pairs as both a sequential and an asy example."""
        sequential, asy = [], []
        for history, target, is_synthetic in pairs:
            history_tokens = " ".join(self.item_tokens[i] for i in history)
            sequential.append({
                "instruction": self.rng.choice(SEQUENTIAL_INSTRUCTIONS),
                "input": history_tokens,
                "output": self.item_tokens[target],
                "task": "sequential",
                "_target": target,
                "_synthetic": is_synthetic,
            })
            asy.append({
                "instruction": self.rng.choice(ASY_INSTRUCTIONS),
                "input": history_tokens,
                "output": self.item_desc[target],
                "task": "asy",
                "_target": target,
                "_synthetic": is_synthetic,
            })
        return sequential, asy

    def build_grounding_examples(self) -> tuple:
        """Build id2name and name2id grounding examples for every item."""
        id2name, name2id = [], []
        for item_id, tokens in self.item_tokens.items():
            blurb = self.item_blurb[item_id]
            output = f"{self.item_desc[item_id]}. {blurb}" if blurb else self.item_desc[item_id]
            id2name.append({
                "instruction": self.rng.choice(ID_TO_NAME_INSTRUCTIONS),
                "input": tokens,
                "output": output,
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

    def _compute_similar_partners(self) -> dict:
        """Compute top co-occurring partner(s) per item, ranked by PMI (not raw co-occurrence count).

        PMI normalizes each pair's co-occurrence by both items' individual
        frequency, so a pair only ranks highly when they co-occur more than
        their popularity alone would predict. Rows tagged `is_synthetic`
        (k-NN-walk-derived) are skipped, since their co-occurrence would be
        circular evidence of similarity.
        """
        logger.info(
            "Computing item co-occurrence (window=%d, min_count=%d)...",
            self.cooccurrence_window, self.min_cooccurrence,
        )
        cooccurrence = Counter()
        item_frequency = Counter()
        n_sequences = 0
        skipped_users = 0
        skipped_synthetic = 0

        for row in self.sequences_df.iter_rows(named=True):
            if self.exclude_long_tail_users and row["is_long_tail_user"]:
                skipped_users += 1
                continue
            if row.get("is_synthetic", False):
                skipped_synthetic += 1
                continue

            items = sorted(set(self._played_sequence(row)[: self.cooccurrence_window]))
            if len(items) < 2:
                continue
            n_sequences += 1
            item_frequency.update(items)
            for a, b in combinations(items, 2):
                cooccurrence[(a, b)] += 1

        logger.info(
            "Found %d co-occurring item pairs across %d qualifying sequences "
            "(skipped %d long-tail users, %d synthetic rows)",
            len(cooccurrence), n_sequences, skipped_users, skipped_synthetic,
        )

        partners: dict = {}
        for (a, b), count in cooccurrence.items():
            if count < self.min_cooccurrence:
                continue
            pmi = math.log((count * n_sequences) / (item_frequency[a] * item_frequency[b]))
            partners.setdefault(a, []).append((b, pmi))
            partners.setdefault(b, []).append((a, pmi))
        return partners

    def build_similar_examples(self, partners: dict) -> List[dict]:
        """Build similar-item examples using semantic-ID input and output."""
        examples = []
        for item_id, candidates in partners.items():
            candidates.sort(key=lambda x: x[1], reverse=True)
            for partner_id, _pmi in candidates[: self.max_similar_per_item]:
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

    def build_nl_similar_examples(self, partners: dict) -> List[dict]:
        """Build nl_similar examples: same co-occurrence ground truth, natural-language input."""
        examples = []
        for item_id, candidates in partners.items():
            candidates.sort(key=lambda x: x[1], reverse=True)
            item_name = self.item_name[item_id]
            for partner_id, _pmi in candidates[: self.max_similar_per_item]:
                query = self.rng.choice(SIMILAR_NL_TEMPLATES).format(item_name=item_name)
                examples.append({
                    "instruction": self.rng.choice(NL_SIMILAR_INSTRUCTIONS),
                    "input": query,
                    "output": self.item_tokens[partner_id],
                    "task": "nl_similar_item",
                    "_target": partner_id,
                })

        logger.info("Built %d raw nl_similar_item examples", len(examples))
        return examples

    @staticmethod
    def _truncate_blurb(about_the_game: Optional[str]) -> str:
        """Return the first sentence of `about_the_game`, hard-capped at MAX_BLURB_WORDS words. Returns "" if empty."""
        if not about_the_game:
            return ""
        first_sentence = re.split(r"(?<=[.!?])\s", about_the_game.strip(), maxsplit=1)[0]
        words = first_sentence.split()
        if len(words) > MAX_BLURB_WORDS:
            return " ".join(words[:MAX_BLURB_WORDS]) + "..."
        return first_sentence

    @staticmethod
    def _natural_genre(genre: str) -> str:
        """Lowercase a catalog genre for natural mid-sentence phrasing, except all-caps acronyms (RPG stays RPG)."""
        return " ".join(word if word.isupper() else word.lower() for word in genre.split())

    # Acronym letters whose spoken name starts with a vowel sound ("an RPG", not "a RPG").
    _VOWEL_SOUND_ACRONYM_LETTERS = set("FHILMNORSX")

    @staticmethod
    def _indefinite_article(genre: str) -> str:
        """Return "a" or "an" as it should precede `genre` in a sentence."""
        first_word = genre.split()[0]
        if first_word.isupper() and len(first_word) > 1:
            return "an" if first_word[0] in AlpacaDatasetBuilder._VOWEL_SOUND_ACRONYM_LETTERS else "a"
        return "an" if first_word[:1].lower() in "aeiou" else "a"

    def build_nl_preference_examples(self) -> List[dict]:
        """Build open-ended genre/genre-combo/genre+category preference queries, several real targets per query type."""
        items_by_genre: dict = defaultdict(list)
        for item_id, genres in self.item_genres.items():
            for genre in genres:
                items_by_genre[genre].append(item_id)

        qualifying_genres = [g for g, items in items_by_genre.items() if len(items) >= MIN_GENRE_ITEM_COUNT]
        logger.info("NL preference: %d genres qualify (>= %d items): %s", len(qualifying_genres), MIN_GENRE_ITEM_COUNT, sorted(qualifying_genres))

        examples = []

        for genre in qualifying_genres:
            natural = self._natural_genre(genre)
            article = self._indefinite_article(natural)
            candidates = self.rng.sample(items_by_genre[genre], min(self.nl_examples_per_genre, len(items_by_genre[genre])))
            for item_id in candidates:
                query = self.rng.choice(GENRE_QUERY_TEMPLATES).format(genre=natural, article=article)
                examples.append({
                    "instruction": self.rng.choice(NL_QUERY_INSTRUCTIONS),
                    "input": query,
                    "output": self.item_tokens[item_id],
                    "task": "nl_preference",
                    "_target": item_id,
                    "criteria": {"genres": [genre], "categories": []},
                })

        for genre1, genre2 in combinations(sorted(qualifying_genres), 2):
            matching = [i for i in items_by_genre[genre1] if genre2 in self.item_genres[i]]
            if len(matching) < MIN_COMBO_ITEM_COUNT:
                continue
            natural1, natural2 = self._natural_genre(genre1), self._natural_genre(genre2)
            article1 = self._indefinite_article(natural1)
            candidates = self.rng.sample(matching, min(self.nl_examples_per_combo, len(matching)))
            for item_id in candidates:
                query = self.rng.choice(GENRE_COMBO_QUERY_TEMPLATES).format(
                    genre1=natural1, genre2=natural2, article1=article1,
                )
                examples.append({
                    "instruction": self.rng.choice(NL_QUERY_INSTRUCTIONS),
                    "input": query,
                    "output": self.item_tokens[item_id],
                    "task": "nl_preference",
                    "_target": item_id,
                    "criteria": {"genres": [genre1, genre2], "categories": []},
                })

        for genre in qualifying_genres:
            natural = self._natural_genre(genre)
            article = self._indefinite_article(natural)
            for raw_category, natural_category in RELEVANT_CATEGORIES.items():
                matching = [i for i in items_by_genre[genre] if raw_category in self.item_categories[i]]
                if len(matching) < MIN_COMBO_ITEM_COUNT:
                    continue
                candidates = self.rng.sample(matching, min(self.nl_examples_per_combo, len(matching)))
                for item_id in candidates:
                    query = self.rng.choice(CATEGORY_QUERY_TEMPLATES).format(
                        genre=natural, category=natural_category, article=article,
                    )
                    examples.append({
                        "instruction": self.rng.choice(NL_QUERY_INSTRUCTIONS),
                        "input": query,
                        "output": self.item_tokens[item_id],
                        "task": "nl_preference",
                        "_target": item_id,
                        "criteria": {"genres": [genre], "categories": [raw_category]},
                    })

        logger.info("Built %d raw nl_preference examples", len(examples))
        return examples

    # ------------------------------------------------------------------
    # Assembly
    # ------------------------------------------------------------------

    def train_val_split_by_group(self, examples: List[dict]) -> tuple:
        """Split by target group rather than by individual example, so near-duplicate oversampled rows don't leak across the split."""
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

    def train_val_split_within_group(self, examples: List[dict]) -> tuple:
        """Split WITHIN each target's group of repeated examples.

        Used for the grounding tasks: every item needs to be groundable, so
        every item's group gets at least one training example and, group
        size permitting, one held-out example testing an unseen phrasing.
        """
        groups = defaultdict(list)
        for ex in examples:
            groups[ex["_target"]].append(ex)

        train, val = [], []
        for group in groups.values():
            group = list(group)
            self.rng.shuffle(group)
            n_val = max(1, round(len(group) * self.val_split)) if len(group) > 1 else 0
            val.extend(group[:n_val])
            train.extend(group[n_val:])
        return train, val

    @staticmethod
    def _exclude_synthetic_from_val(train: List[dict], val: List[dict]) -> tuple:
        """Move any `_synthetic`-tagged example out of val and into train, so eval only measures real user behavior."""
        leaked_synthetic = [ex for ex in val if ex.get("_synthetic")]
        if not leaked_synthetic:
            return train, val
        val = [ex for ex in val if not ex.get("_synthetic")]
        train = train + leaked_synthetic
        return train, val

    def build_all(self) -> dict:
        """Run the full pipeline: load data, build per-task examples, rebalance, split, and write JSONL."""
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

        partners = self._compute_similar_partners()
        similar = self.build_similar_examples(partners)
        similar = self._rebalance_by_target(
            similar, lambda ex: ex["_target"], self.similar_target_floor, self.similar_target_ceiling, SIMILAR_INSTRUCTIONS,
        )
        nl_similar = self.build_nl_similar_examples(partners)
        nl_similar = self._rebalance_by_target(
            nl_similar, lambda ex: ex["_target"], self.similar_target_floor, self.similar_target_ceiling, NL_SIMILAR_INSTRUCTIONS,
        )

        nl_preference = self.build_nl_preference_examples()

        relatedness = build_relatedness_examples(
            self.item_codes, self.item_tokens, self.relatedness_examples_per_item, self.rng,
        )
        logger.info("Built %d relatedness examples", len(relatedness))

        recommendation_tasks = self._cap_total_exposure_across_tasks(
            {
                "sequential": sequential, "asy": asy,
                "similar_item": similar, "nl_similar_item": nl_similar,
                "nl_preference": nl_preference,
            },
            self.max_total_recommendation_exposure,
        )

        tasks = {
            "sequential": recommendation_tasks["sequential"],
            "asy": recommendation_tasks["asy"],
            "grounding_id2name": id2name, "grounding_name2id": name2id,
            "similar_item": recommendation_tasks["similar_item"],
            "nl_similar_item": recommendation_tasks["nl_similar_item"],
            "nl_preference": recommendation_tasks["nl_preference"],
            "relatedness": relatedness,
        }

        split_fn_by_task = {
            "grounding_id2name": self.train_val_split_within_group,
            "grounding_name2id": self.train_val_split_within_group,
        }

        train_all, val_all = [], []
        for name, examples in tasks.items():
            split_fn = split_fn_by_task.get(name, self.train_val_split_by_group)
            train, val = split_fn(examples)
            train, val = self._exclude_synthetic_from_val(train, val)
            train_all.extend(train)
            val_all.extend(val)
            logger.info("%s: %d train, %d val", name, len(train), len(val))

        self.rng.shuffle(train_all)
        self.rng.shuffle(val_all)

        for ex in train_all + val_all:
            for key in [k for k in ex if k.startswith("_")]:
                del ex[key]

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
        """Apply floor/ceiling rebalancing to raw (history, target) pairs, before sequential/asy render them separately."""
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
        """Write `examples` as JSONL to `path`."""
        with open(path, "w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        logger.info("Wrote %d examples to %s", len(examples), path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sequences-path", type=Path, default=None,
        help="Defaults to data/clean_user_sequences.parquet. Point at data/combined_user_sequences.parquet "
             "to include the synthetic sequential top-up.",
    )
    args = parser.parse_args()

    config = RQVAEConfig()
    builder = AlpacaDatasetBuilder(config, sequences_path=args.sequences_path)
    builder.build_all()