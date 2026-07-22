"""Unit tests for AlpacaDatasetBuilder's pure in-memory logic: floor/ceiling
rebalancing, group-preserving train/val splitting, and special-token
generation. None of these call load_data(), so no parquet files are needed
-- they operate on hand-built example lists."""

import sys
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
