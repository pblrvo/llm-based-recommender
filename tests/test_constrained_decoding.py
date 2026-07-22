"""Unit tests for the pure-logic pieces of constrained_decoding.py: the
Trie, semantic-ID/description formatting, sid-code parsing, hierarchical
prefix-accuracy scoring, Recall@K/NDCG@K, and the prefix_allowed_tokens_fn
used to drive constrained generation. No model, tokenizer download, or GPU
required."""

import math
import sys
from pathlib import Path

import polars as pl
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from constrained_decoding import (
    Trie,
    build_name_trie,
    build_sid_trie,
    hierarchical_match,
    item_description,
    make_prefix_allowed_tokens_fn,
    ndcg_at_k,
    parse_sid_codes,
    recall_at_k,
    semantic_id_to_tokens,
)


class WordTokenizer:
    """Minimal fake tokenizer: one token id per whitespace-separated word,
    assigned in first-seen order. Enough to exercise Trie construction
    without needing a real HF tokenizer/model download."""

    def __init__(self):
        self.vocab = {}

    def _id_for(self, word: str) -> int:
        return self.vocab.setdefault(word, len(self.vocab))

    def __call__(self, text: str, add_special_tokens: bool = False):
        return {"input_ids": [self._id_for(w) for w in text.split()]}


# ---------------------------------------------------------------------
# semantic_id_to_tokens / item_description
# ---------------------------------------------------------------------


def test_semantic_id_to_tokens_formats_all_levels_in_order():
    assert semantic_id_to_tokens([12, 34, 56, 0]) == (
        "<|sid_start|><|sid_L0_12|><|sid_L1_34|><|sid_L2_56|><|sid_L3_0|><|sid_end|>"
    )


def test_semantic_id_to_tokens_handles_arbitrary_length():
    assert semantic_id_to_tokens([1]) == "<|sid_start|><|sid_L0_1|><|sid_end|>"


def test_item_description_joins_name_and_genres():
    assert item_description("Half-Life 2", "Action,Adventure") == "Half-Life 2 — Action, Adventure"


def test_item_description_omits_dash_when_genres_missing():
    assert item_description("Untitled Goose Game", None) == "Untitled Goose Game"
    assert item_description("Untitled Goose Game", "") == "Untitled Goose Game"


# ---------------------------------------------------------------------
# parse_sid_codes
# ---------------------------------------------------------------------


def test_parse_sid_codes_extracts_all_four_levels_in_order():
    text = "<|sid_start|><|sid_L0_12|><|sid_L1_34|><|sid_L2_56|><|sid_L3_0|><|sid_end|>"
    assert parse_sid_codes(text) == [12, 34, 56, 0]


def test_parse_sid_codes_order_independent_of_token_order_in_text():
    # Levels are matched by their embedded index, not text position.
    text = "<|sid_L3_9|><|sid_L1_2|><|sid_L0_1|><|sid_L2_3|>"
    assert parse_sid_codes(text) == [1, 2, 3, 9]


def test_parse_sid_codes_rejects_missing_level():
    text = "<|sid_start|><|sid_L0_1|><|sid_L1_2|><|sid_L2_3|><|sid_end|>"  # no L3
    assert parse_sid_codes(text) is None


def test_parse_sid_codes_rejects_duplicate_level():
    text = "<|sid_L0_1|><|sid_L0_2|><|sid_L1_3|><|sid_L2_4|>"  # L0 twice, no L3
    assert parse_sid_codes(text) is None


def test_parse_sid_codes_rejects_garbage_text():
    assert parse_sid_codes("The Witcher 2 — RPG") is None
    assert parse_sid_codes("") is None


# ---------------------------------------------------------------------
# hierarchical_match
# ---------------------------------------------------------------------


def test_hierarchical_match_exact_match_all_true():
    sid = semantic_id_to_tokens([1, 2, 3, 0])
    result = hierarchical_match(sid, sid)
    assert result == {"valid_format": True, "l0": True, "l01": True, "l012": True, "l0123": True}


def test_hierarchical_match_partial_prefix_agreement():
    predicted = semantic_id_to_tokens([1, 2, 99, 0])
    expected = semantic_id_to_tokens([1, 2, 3, 0])
    result = hierarchical_match(predicted, expected)
    assert result["l0"] is True
    assert result["l01"] is True
    assert result["l012"] is False  # diverges at level 2
    assert result["l0123"] is False


def test_hierarchical_match_wrong_from_first_level():
    predicted = semantic_id_to_tokens([9, 2, 3, 0])
    expected = semantic_id_to_tokens([1, 2, 3, 0])
    result = hierarchical_match(predicted, expected)
    assert result == {"valid_format": True, "l0": False, "l01": False, "l012": False, "l0123": False}


def test_hierarchical_match_malformed_prediction_is_all_false():
    result = hierarchical_match("not a valid sid", semantic_id_to_tokens([1, 2, 3, 0]))
    assert result["valid_format"] is False
    assert result["l0"] is False and result["l0123"] is False


# ---------------------------------------------------------------------
# Trie
# ---------------------------------------------------------------------


def test_trie_insert_and_children_of_root():
    trie = Trie()
    trie.insert([1, 2, 3])
    allowed_at_root = [tok for tok in trie.children_of([]) if tok != Trie.END]
    assert allowed_at_root == [1]


def test_trie_children_of_unknown_prefix_returns_none():
    trie = Trie()
    trie.insert([1, 2, 3])
    assert trie.children_of([9]) is None


def test_trie_end_marker_present_only_at_valid_stopping_point():
    trie = Trie()
    trie.insert([1, 2])
    assert Trie.END not in trie.children_of([1])  # mid-sequence: not a stopping point
    assert Trie.END in trie.children_of([1, 2])  # full sequence: valid stop


def test_trie_supports_strict_prefix_of_another_entry():
    """[1] is a strict prefix of [1, 2] -- both must remain valid endpoints."""
    trie = Trie()
    trie.insert([1])
    trie.insert([1, 2])
    assert Trie.END in trie.children_of([1])
    assert 2 in trie.children_of([1])
    assert Trie.END in trie.children_of([1, 2])


def test_trie_branches_on_divergent_sequences():
    trie = Trie()
    trie.insert([1, 2])
    trie.insert([1, 3])
    allowed = [tok for tok in trie.children_of([1]) if tok != Trie.END]
    assert sorted(allowed) == [2, 3]


# ---------------------------------------------------------------------
# make_prefix_allowed_tokens_fn
# ---------------------------------------------------------------------


def test_prefix_allowed_tokens_fn_offers_only_trie_children():
    trie = Trie()
    trie.insert([10, 20, 30])
    fn = make_prefix_allowed_tokens_fn(trie, prompt_len=5, eos_token_id=999)

    prompt = [0, 0, 0, 0, 0]  # first prompt_len tokens are ignored by the fn
    allowed_at_start = fn(0, torch.tensor(prompt))
    assert allowed_at_start == [10]

    allowed_after_one_token = fn(0, torch.tensor(prompt + [10]))
    assert allowed_after_one_token == [20]


def test_prefix_allowed_tokens_fn_appends_eos_at_valid_stop():
    trie = Trie()
    trie.insert([10, 20])
    fn = make_prefix_allowed_tokens_fn(trie, prompt_len=0, eos_token_id=999)

    allowed = fn(0, torch.tensor([10, 20]))
    assert set(allowed) == {999}  # only EOS is valid once the sequence is complete


def test_prefix_allowed_tokens_fn_falls_back_to_eos_when_desynced():
    trie = Trie()
    trie.insert([10, 20])
    fn = make_prefix_allowed_tokens_fn(trie, prompt_len=0, eos_token_id=999)

    # A generated sequence that was never a valid trie path.
    allowed = fn(0, torch.tensor([77, 78]))
    assert allowed == [999]


# ---------------------------------------------------------------------
# build_sid_trie / build_name_trie (fake tokenizer, no model download)
# ---------------------------------------------------------------------


@pytest.fixture
def tiny_catalog():
    return pl.DataFrame({
        "id": [1, 2],
        "semantic_ids": [[1, 2, 3, 0], [4, 5, 6, 0]],
        "Name": ["Half-Life 2", "Portal 2"],
        "Genres": ["Action", "Puzzle,Comedy"],
    })


def test_build_sid_trie_accepts_every_catalog_semantic_id(tiny_catalog):
    tokenizer = WordTokenizer()
    trie = build_sid_trie(tokenizer, tiny_catalog)

    for row in tiny_catalog.iter_rows(named=True):
        token_ids = tokenizer(semantic_id_to_tokens(row["semantic_ids"]))["input_ids"]
        node = trie.children_of(token_ids)
        assert node is not None and Trie.END in node


def test_build_name_trie_accepts_every_catalog_description(tiny_catalog):
    tokenizer = WordTokenizer()
    trie = build_name_trie(tokenizer, tiny_catalog)

    for row in tiny_catalog.iter_rows(named=True):
        desc = item_description(row["Name"], row["Genres"])
        token_ids = tokenizer(desc)["input_ids"]
        node = trie.children_of(token_ids)
        assert node is not None and Trie.END in node


# ---------------------------------------------------------------------
# recall_at_k / ndcg_at_k
# ---------------------------------------------------------------------


def test_recall_at_k_hit_within_k():
    assert recall_at_k(["a", "b", "c"], "b", k=3) == 1.0


def test_recall_at_k_miss_outside_k():
    assert recall_at_k(["a", "b", "c", "target"], "target", k=3) == 0.0


def test_recall_at_k_miss_not_present_at_all():
    assert recall_at_k(["a", "b", "c"], "target", k=10) == 0.0


def test_recall_at_k_handles_fewer_candidates_than_k():
    assert recall_at_k(["a"], "a", k=10) == 1.0
    assert recall_at_k(["a"], "b", k=10) == 0.0


def test_recall_at_k_hit_exactly_at_boundary():
    # target is the k-th candidate (index k-1) -- must still count as a hit.
    assert recall_at_k(["a", "b", "c"], "c", k=3) == 1.0
    assert recall_at_k(["a", "b", "c", "d"], "d", k=3) == 0.0  # one past the boundary


def test_ndcg_at_k_rank_1_is_perfect_score():
    assert ndcg_at_k(["target", "b", "c"], "target", k=3) == pytest.approx(1.0)


def test_ndcg_at_k_decreases_with_rank():
    ndcg_rank_1 = ndcg_at_k(["target", "b", "c"], "target", k=3)
    ndcg_rank_2 = ndcg_at_k(["a", "target", "c"], "target", k=3)
    ndcg_rank_3 = ndcg_at_k(["a", "b", "target"], "target", k=3)
    assert ndcg_rank_1 > ndcg_rank_2 > ndcg_rank_3 > 0


def test_ndcg_at_k_matches_log2_formula_at_each_rank():
    candidates = ["a", "b", "target", "d"]
    assert ndcg_at_k(candidates, "target", k=4) == pytest.approx(1 / math.log2(3 + 1))


def test_ndcg_at_k_zero_when_outside_k():
    assert ndcg_at_k(["a", "b", "c", "target"], "target", k=2) == 0.0


def test_ndcg_at_k_zero_when_absent():
    assert ndcg_at_k(["a", "b", "c"], "target", k=3) == 0.0
