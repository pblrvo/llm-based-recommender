"""Unit tests for AlpacaDatasetBuilder's pure in-memory logic: floor/ceiling
rebalancing, group-preserving train/val splitting, and special-token
generation. None of these call load_data(), so no parquet files are needed
-- they operate on hand-built example lists."""

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from build_finetune_dataset import SID_END, SID_START, AlpacaDatasetBuilder
from config import RQVAEConfig


def make_builder(seed=0, **overrides) -> AlpacaDatasetBuilder:
    config = RQVAEConfig(codebook_quantization_levels=3, codebook_size=8)
    return AlpacaDatasetBuilder(config, seed=seed, **overrides)


def make_examples(target_counts: dict, task="sequential") -> list:
    """target_counts: {target_key: how many examples to generate for it}."""
    examples = []
    for target, count in target_counts.items():
        for i in range(count):
            examples.append({
                "instruction": "original instruction",
                "input": f"input-{target}-{i}",
                "output": f"output-{target}",
                "task": task,
                "_target": target,
            })
    return examples


# ---------------------------------------------------------------------
# get_special_tokens
# ---------------------------------------------------------------------


def test_get_special_tokens_count_and_markers():
    builder = make_builder()
    tokens = builder.get_special_tokens()

    # codebook_quantization_levels=3 -> 4 levels (L0..L3, +1 for disambiguation)
    # at codebook_size=8 each, plus the 2 start/end markers.
    assert len(tokens) == 2 + 4 * 8
    assert SID_START in tokens
    assert SID_END in tokens
    assert "<|sid_L0_0|>" in tokens
    assert "<|sid_L3_7|>" in tokens  # last level, last code
    assert "<|sid_L4_0|>" not in tokens  # only 4 levels exist


def test_get_special_tokens_are_all_unique():
    builder = make_builder()
    tokens = builder.get_special_tokens()
    assert len(tokens) == len(set(tokens))


# ---------------------------------------------------------------------
# semantic_id_to_tokens
# ---------------------------------------------------------------------


def test_semantic_id_to_tokens_matches_expected_format():
    builder = make_builder()
    assert builder.semantic_id_to_tokens([1, 2, 3, 0]) == (
        f"{SID_START}<|sid_L0_1|><|sid_L1_2|><|sid_L2_3|><|sid_L3_0|>{SID_END}"
    )


# ---------------------------------------------------------------------
# _rebalance_by_target: ceiling (subsampling)
# ---------------------------------------------------------------------


def test_rebalance_caps_oversized_groups_at_ceiling():
    builder = make_builder()
    examples = make_examples({"a": 20, "b": 3})
    result = builder._rebalance_by_target(examples, lambda ex: ex["_target"], floor=1, ceiling=10)

    counts = {"a": 0, "b": 0}
    for ex in result:
        counts[ex["_target"]] += 1
    assert counts["a"] == 10  # capped
    assert counts["b"] == 3  # untouched, already within range


def test_rebalance_ceiling_subsample_keeps_only_original_examples():
    """Subsampling should never invent new content, only select a subset."""
    builder = make_builder()
    examples = make_examples({"a": 20})
    result = builder._rebalance_by_target(examples, lambda ex: ex["_target"], floor=1, ceiling=5)
    original_inputs = {ex["input"] for ex in examples}
    for ex in result:
        assert ex["input"] in original_inputs


# ---------------------------------------------------------------------
# _rebalance_by_target: floor (oversampling)
# ---------------------------------------------------------------------


def test_rebalance_oversamples_undersized_groups_to_floor():
    builder = make_builder()
    examples = make_examples({"a": 2})
    result = builder._rebalance_by_target(examples, lambda ex: ex["_target"], floor=7, ceiling=100)
    assert len(result) == 7


def test_rebalance_oversampling_preserves_all_originals():
    builder = make_builder()
    examples = make_examples({"a": 2})
    result = builder._rebalance_by_target(examples, lambda ex: ex["_target"], floor=7, ceiling=100)
    original_inputs = {ex["input"] for ex in examples}
    result_inputs = {ex["input"] for ex in result}
    assert original_inputs <= result_inputs  # every original example is still present


def test_rebalance_oversampling_varies_instruction_from_pool():
    builder = make_builder()
    examples = make_examples({"a": 1})
    pool = ["instruction A", "instruction B", "instruction C"]
    result = builder._rebalance_by_target(examples, lambda ex: ex["_target"], floor=20, ceiling=100, instruction_pool=pool)

    # The single original example keeps "original instruction"; all 19 clones
    # should have been re-rolled from the pool instead.
    instructions_used = {ex["instruction"] for ex in result}
    assert instructions_used <= ({"original instruction"} | set(pool))
    assert any(ex["instruction"] in pool for ex in result)


def test_rebalance_without_instruction_pool_keeps_original_instruction_on_clones():
    builder = make_builder()
    examples = make_examples({"a": 1})
    result = builder._rebalance_by_target(examples, lambda ex: ex["_target"], floor=5, ceiling=100)
    assert all(ex["instruction"] == "original instruction" for ex in result)


def test_rebalance_leaves_in_range_groups_unchanged_in_count():
    builder = make_builder()
    examples = make_examples({"a": 5})
    result = builder._rebalance_by_target(examples, lambda ex: ex["_target"], floor=2, ceiling=10)
    assert len(result) == 5


def test_rebalance_every_group_lands_within_floor_and_ceiling():
    builder = make_builder()
    examples = make_examples({"tiny": 1, "just_right": 6, "huge": 50})
    result = builder._rebalance_by_target(examples, lambda ex: ex["_target"], floor=5, ceiling=15)

    counts = {}
    for ex in result:
        counts[ex["_target"]] = counts.get(ex["_target"], 0) + 1
    for target, count in counts.items():
        assert 5 <= count <= 15, f"{target} landed at {count}, outside [5, 15]"


def test_rebalance_is_deterministic_given_same_seed():
    examples = make_examples({"a": 1, "b": 20})
    result_1 = make_builder(seed=42)._rebalance_by_target(examples, lambda ex: ex["_target"], floor=5, ceiling=10)
    result_2 = make_builder(seed=42)._rebalance_by_target(examples, lambda ex: ex["_target"], floor=5, ceiling=10)
    assert [ex["input"] for ex in result_1] == [ex["input"] for ex in result_2]


# ---------------------------------------------------------------------
# _cap_total_exposure_across_tasks
# ---------------------------------------------------------------------


def test_cap_total_exposure_caps_combined_count_across_tasks():
    builder = make_builder()
    # target "popular" hits its own ceiling in three separate tasks:
    # 20 + 20 + 20 = 60 combined, well above a cap of 40.
    task_examples = {
        "sequential": make_examples({"popular": 20, "rare": 2}, task="sequential"),
        "similar_item": make_examples({"popular": 20}, task="similar_item"),
        "nl_similar_item": make_examples({"popular": 20}, task="nl_similar_item"),
    }
    result = builder._cap_total_exposure_across_tasks(task_examples, max_total=40)

    total_popular = sum(
        1 for examples in result.values() for ex in examples if ex["_target"] == "popular"
    )
    total_rare = sum(
        1 for examples in result.values() for ex in examples if ex["_target"] == "rare"
    )
    assert total_popular == 40
    assert total_rare == 2  # untouched, already under the cap


def test_cap_total_exposure_leaves_items_under_cap_untouched():
    builder = make_builder()
    task_examples = {
        "sequential": make_examples({"a": 5}, task="sequential"),
        "similar_item": make_examples({"a": 3}, task="similar_item"),
    }
    result = builder._cap_total_exposure_across_tasks(task_examples, max_total=40)
    assert len(result["sequential"]) == 5
    assert len(result["similar_item"]) == 3


def test_cap_total_exposure_preserves_task_keys_even_when_empty():
    builder = make_builder()
    task_examples = {
        "sequential": make_examples({"popular": 50}, task="sequential"),
        "nl_preference": [],
    }
    result = builder._cap_total_exposure_across_tasks(task_examples, max_total=10)
    assert set(result.keys()) == {"sequential", "nl_preference"}
    assert result["nl_preference"] == []
    assert len(result["sequential"]) == 10


def test_cap_total_exposure_does_not_affect_other_targets():
    builder = make_builder()
    task_examples = {
        "sequential": make_examples({"popular": 60, "other": 4}, task="sequential"),
    }
    result = builder._cap_total_exposure_across_tasks(task_examples, max_total=40)
    counts = Counter(ex["_target"] for ex in result["sequential"])
    assert counts["popular"] == 40
    assert counts["other"] == 4


# ---------------------------------------------------------------------
# _rebalance_pairs_by_target (history, target) tuples
# ---------------------------------------------------------------------


def test_rebalance_pairs_caps_and_floors_like_example_version():
    builder = make_builder()
    pairs = [(["h"], "popular")] * 30 + [(["h"], "rare")] * 1
    result = builder._rebalance_pairs_by_target(pairs, floor=5, ceiling=10)

    counts = {"popular": 0, "rare": 0}
    for _, target in result:
        counts[target] += 1
    assert counts["popular"] == 10
    assert counts["rare"] == 5


# ---------------------------------------------------------------------
# train_val_split_by_group
# ---------------------------------------------------------------------


def test_train_val_split_keeps_each_group_entirely_on_one_side():
    builder = make_builder(val_split=0.5)
    examples = make_examples({"a": 3, "b": 3, "c": 3, "d": 3})
    train, val = builder.train_val_split_by_group(examples)

    train_targets = {ex["_target"] for ex in train}
    val_targets = {ex["_target"] for ex in val}
    assert train_targets.isdisjoint(val_targets)  # no group split across both
    assert train_targets | val_targets == {"a", "b", "c", "d"}


def test_train_val_split_covers_every_example_exactly_once():
    builder = make_builder(val_split=0.3)
    examples = make_examples({"a": 4, "b": 4, "c": 4})
    train, val = builder.train_val_split_by_group(examples)
    assert len(train) + len(val) == len(examples)


def test_train_val_split_always_reserves_at_least_one_group_for_val():
    """max(1, int(n_groups * val_split)) -- even a tiny val_split shouldn't
    produce an empty validation set as long as groups exist."""
    builder = make_builder(val_split=0.01)
    examples = make_examples({"a": 2, "b": 2})
    train, val = builder.train_val_split_by_group(examples)
    assert len(val) > 0


def test_train_val_split_single_group_goes_entirely_to_val():
    builder = make_builder(val_split=0.5)
    examples = make_examples({"only": 5})
    train, val = builder.train_val_split_by_group(examples)
    assert len(train) == 0
    assert len(val) == 5


# ---------------------------------------------------------------------
# train_val_split_within_group (grounding tasks: every item must be
# trained on at least once -- see the function's own docstring for why)
# ---------------------------------------------------------------------


def test_within_group_split_every_group_appears_in_both_train_and_val():
    builder = make_builder(val_split=0.1)
    examples = make_examples({"a": 10, "b": 10, "c": 10})
    train, val = builder.train_val_split_within_group(examples)

    train_targets = {ex["_target"] for ex in train}
    val_targets = {ex["_target"] for ex in val}
    assert train_targets == {"a", "b", "c"}
    assert val_targets == {"a", "b", "c"}  # every item held out at least once too


def test_within_group_split_covers_every_example_exactly_once():
    builder = make_builder(val_split=0.1)
    examples = make_examples({"a": 10, "b": 7})
    train, val = builder.train_val_split_within_group(examples)
    assert len(train) + len(val) == len(examples)
    assert {ex["input"] for ex in train} | {ex["input"] for ex in val} == {ex["input"] for ex in examples}
    assert {ex["input"] for ex in train} & {ex["input"] for ex in val} == set()  # no example in both


def test_within_group_split_group_of_one_goes_entirely_to_train():
    """A group that can't be split without leaving nothing to train on
    keeps its single example in train -- an ungroundable item would be
    worse than an untested one."""
    builder = make_builder(val_split=0.5)
    examples = make_examples({"only": 1})
    train, val = builder.train_val_split_within_group(examples)
    assert len(train) == 1
    assert len(val) == 0


def test_within_group_split_reserves_at_least_one_val_example_per_group():
    builder = make_builder(val_split=0.01)  # tiny fraction, would round to 0 without the floor
    examples = make_examples({"a": 10})
    train, val = builder.train_val_split_within_group(examples)
    assert len(val) >= 1
    assert len(train) >= 1


# ---------------------------------------------------------------------
# NL preference / NL similar-item queries
# ---------------------------------------------------------------------


def test_natural_genre_lowercases_normal_words():
    assert AlpacaDatasetBuilder._natural_genre("Action") == "action"
    assert AlpacaDatasetBuilder._natural_genre("Free To Play") == "free to play"


def test_natural_genre_preserves_acronyms():
    assert AlpacaDatasetBuilder._natural_genre("RPG") == "RPG"


def test_indefinite_article_vowel_sound_words():
    assert AlpacaDatasetBuilder._indefinite_article("action") == "an"
    assert AlpacaDatasetBuilder._indefinite_article("Adventure") == "an"
    assert AlpacaDatasetBuilder._indefinite_article("Indie") == "an"


def test_indefinite_article_consonant_sound_words():
    assert AlpacaDatasetBuilder._indefinite_article("racing") == "a"
    assert AlpacaDatasetBuilder._indefinite_article("Casual") == "a"
    assert AlpacaDatasetBuilder._indefinite_article("Strategy") == "a"


def test_indefinite_article_acronym_judged_by_spoken_letter_name():
    # "RPG" is pronounced "are-pee-jee" -- "an RPG", not "a RPG", even
    # though R is a consonant letter.
    assert AlpacaDatasetBuilder._indefinite_article("RPG") == "an"


# ---------------------------------------------------------------------
# _truncate_blurb / grounding_id2name description enrichment
# ---------------------------------------------------------------------


def test_truncate_blurb_returns_empty_string_for_missing_input():
    assert AlpacaDatasetBuilder._truncate_blurb(None) == ""
    assert AlpacaDatasetBuilder._truncate_blurb("") == ""


def test_truncate_blurb_keeps_short_first_sentence_whole():
    text = "A fast-paced racing game with online multiplayer. More text here that should be dropped."
    assert AlpacaDatasetBuilder._truncate_blurb(text) == "A fast-paced racing game with online multiplayer."


def test_truncate_blurb_caps_long_first_sentence_at_max_words():
    import build_finetune_dataset as bfd
    long_sentence = " ".join(f"word{i}" for i in range(bfd.MAX_BLURB_WORDS + 10)) + "."
    result = AlpacaDatasetBuilder._truncate_blurb(long_sentence)
    assert result.endswith("...")
    assert len(result[:-3].split()) == bfd.MAX_BLURB_WORDS


def test_truncate_blurb_handles_text_with_no_sentence_break():
    text = "just one long run-on phrase with no terminal punctuation at all here"
    assert AlpacaDatasetBuilder._truncate_blurb(text) == text


def test_grounding_id2name_appends_blurb_but_name2id_and_item_desc_stay_short():
    builder = make_builder()
    builder.item_tokens = {"a1": "<sid-a1>"}
    builder.item_name = {"a1": "Game A"}
    builder.item_desc = {"a1": "Game A — Action, Indie"}
    builder.item_blurb = {"a1": "A short punchy summary."}

    id2name, name2id = builder.build_grounding_examples()

    assert id2name[0]["output"] == "Game A — Action, Indie. A short punchy summary."
    assert name2id[0]["output"] == "<sid-a1>"
    assert name2id[0]["input"] == "Game A"
    # item_desc itself (shared with asy) is untouched by the blurb.
    assert builder.item_desc["a1"] == "Game A — Action, Indie"


def test_grounding_id2name_omits_blurb_separator_when_blurb_is_empty():
    builder = make_builder()
    builder.item_tokens = {"a1": "<sid-a1>"}
    builder.item_name = {"a1": "Game A"}
    builder.item_desc = {"a1": "Game A — Action, Indie"}
    builder.item_blurb = {"a1": ""}

    id2name, _ = builder.build_grounding_examples()
    assert id2name[0]["output"] == "Game A — Action, Indie"


def make_catalog_builder(genre_items: dict, category_items: dict = None, seed=0):
    """A builder with hand-populated item_genres/item_categories/item_tokens/
    item_name, bypassing load_data() entirely -- genre_items maps genre name
    -> list of item ids that should carry that genre."""
    builder = make_builder(seed=seed)
    all_ids = {i for ids in genre_items.values() for i in ids}
    for item_id in all_ids:
        builder.item_tokens[item_id] = f"<sid-{item_id}>"
        builder.item_name[item_id] = f"Game {item_id}"
        builder.item_genres[item_id] = set()
        builder.item_categories[item_id] = set()
    for genre, ids in genre_items.items():
        for item_id in ids:
            builder.item_genres[item_id].add(genre)
    for category, ids in (category_items or {}).items():
        for item_id in ids:
            builder.item_categories[item_id].add(category)
    return builder


def test_nl_preference_single_genre_query_only_targets_matching_items():
    import build_finetune_dataset as bfd
    original_min = bfd.MIN_GENRE_ITEM_COUNT
    bfd.MIN_GENRE_ITEM_COUNT = 3
    try:
        builder = make_catalog_builder({"Action": [f"a{i}" for i in range(5)], "Racing": [f"r{i}" for i in range(1)]})
        examples = builder.build_nl_preference_examples()
    finally:
        bfd.MIN_GENRE_ITEM_COUNT = original_min

    assert len(examples) > 0
    for ex in examples:
        assert ex["task"] == "nl_preference"
        assert ex["_target"] in {f"a{i}" for i in range(5)}  # Racing had too few items to qualify
        assert ex["output"] == builder.item_tokens[ex["_target"]]


def test_nl_preference_genre_combo_requires_minimum_overlap():
    import build_finetune_dataset as bfd
    original_genre_min, original_combo_min = bfd.MIN_GENRE_ITEM_COUNT, bfd.MIN_COMBO_ITEM_COUNT
    bfd.MIN_GENRE_ITEM_COUNT = 3
    bfd.MIN_COMBO_ITEM_COUNT = 2
    try:
        # Action={a0,a1,a2}, RPG={a1,r0,r1} -- overlap is just {a1} (1 item),
        # below MIN_COMBO_ITEM_COUNT=2, so the combo must be skipped entirely:
        # expect exactly 3 (Action) + 3 (RPG) = 6 single-genre examples, 0 combo.
        builder = make_catalog_builder({"Action": ["a0", "a1", "a2"], "RPG": ["a1", "r0", "r1"]})
        builder.nl_examples_per_genre = 100  # no cap in play, so the count below is exact
        examples = builder.build_nl_preference_examples()
        assert len(examples) == 6
    finally:
        bfd.MIN_GENRE_ITEM_COUNT, bfd.MIN_COMBO_ITEM_COUNT = original_genre_min, original_combo_min


def test_nl_preference_genre_combo_generated_when_overlap_is_sufficient():
    import build_finetune_dataset as bfd
    original_genre_min, original_combo_min = bfd.MIN_GENRE_ITEM_COUNT, bfd.MIN_COMBO_ITEM_COUNT
    bfd.MIN_GENRE_ITEM_COUNT = 3
    bfd.MIN_COMBO_ITEM_COUNT = 2
    try:
        # Action={a0,a1,a2,a3}, RPG={a0,a1,r0} -- overlap {a0,a1} (2 items),
        # meets MIN_COMBO_ITEM_COUNT=2: expect 4 (Action) + 3 (RPG) + 2 (combo) = 9.
        builder = make_catalog_builder({"Action": ["a0", "a1", "a2", "a3"], "RPG": ["a0", "a1", "r0"]})
        builder.nl_examples_per_genre = 100
        builder.nl_examples_per_combo = 100
        examples = builder.build_nl_preference_examples()
        assert len(examples) == 9
    finally:
        bfd.MIN_GENRE_ITEM_COUNT, bfd.MIN_COMBO_ITEM_COUNT = original_genre_min, original_combo_min


def test_nl_preference_respects_examples_per_genre_cap():
    import build_finetune_dataset as bfd
    original_min = bfd.MIN_GENRE_ITEM_COUNT
    bfd.MIN_GENRE_ITEM_COUNT = 3
    try:
        builder = make_catalog_builder({"Action": [f"a{i}" for i in range(50)]})
        builder.nl_examples_per_genre = 5
        examples = builder.build_nl_preference_examples()
    finally:
        bfd.MIN_GENRE_ITEM_COUNT = original_min

    single_genre_examples = [ex for ex in examples]
    assert len(single_genre_examples) == 5  # capped, not all 50 matching items used


def test_compute_similar_partners_and_nl_similar_examples():
    import polars as pl

    builder = make_builder(seed=0, min_cooccurrence=1, max_similar_per_item=5)
    builder.item_tokens = {"x": "<sid-x>", "y": "<sid-y>", "z": "<sid-z>"}
    builder.item_name = {"x": "Game X", "y": "Game Y", "z": "Game Z"}
    builder.sequences_df = pl.DataFrame({
        "item_sequence": [["x", "y", "z"], ["x", "y"]],
        "is_long_tail_user": [False, False],
    })

    partners = builder._compute_similar_partners()
    assert "x" in partners and "y" in partners

    nl_examples = builder.build_nl_similar_examples(partners)
    assert len(nl_examples) > 0
    item_names = set(builder.item_name.values())
    for ex in nl_examples:
        assert ex["task"] == "nl_similar_item"
        assert any(name in ex["input"] for name in item_names)  # input references a real item's name
        assert ex["output"] in builder.item_tokens.values()
