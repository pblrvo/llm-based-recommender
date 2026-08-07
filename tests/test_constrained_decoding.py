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
    build_name_lookup,
    build_name_trie,
    build_sid_criteria_lookup,
    build_sid_trie,
    criteria_ndcg_at_k,
    criteria_satisfied_at_k,
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
        "Categories": [None, "Single-player,Multi-player"],
        "About the game": ["A first-person shooter set in City 17. More detail follows here.", None],
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
        desc = item_description(row["Name"], row["Genres"], row["About the game"])
        token_ids = tokenizer(desc)["input_ids"]
        node = trie.children_of(token_ids)
        assert node is not None and Trie.END in node


def test_build_name_trie_includes_the_blurb_not_just_name_and_genres(tiny_catalog):
    # Half-Life 2 has a real "About the game" blurb -- build_name_trie's
    # entry for it must be the full blurb-included description, not the
    # plain "Name — Genres" short form, since that's what
    # grounding_id2name's real training target looks like (see
    # build_finetune_dataset.py's build_grounding_examples). A trie built
    # from the short form alone (the pre-fix behavior) would only ever
    # recognize "Half-Life 2 — Action" as complete, never the real target.
    short_form = item_description("Half-Life 2", "Action")
    full_desc = item_description("Half-Life 2", "Action", tiny_catalog.row(0, named=True)["About the game"])
    assert full_desc != short_form
    assert full_desc.startswith(short_form + ".")

    tokenizer = WordTokenizer()
    trie = build_name_trie(tokenizer, tiny_catalog)
    node = trie.children_of(tokenizer(full_desc)["input_ids"])
    assert node is not None and Trie.END in node


def test_build_name_trie_also_accepts_the_short_form_for_asy(tiny_catalog):
    # asy's real training target is the plain "Name — Genres" short form,
    # never the blurb -- the trie must accept that as a valid stopping
    # point too, not just the long form grounding_id2name produces (see
    # build_name_trie's docstring). Without this, constraining asy's
    # generation with this trie would force it past where it's actually
    # trained to stop.
    tokenizer = WordTokenizer()
    trie = build_name_trie(tokenizer, tiny_catalog)

    short_form = item_description("Half-Life 2", "Action")
    node = trie.children_of(tokenizer(short_form)["input_ids"])
    assert node is not None and Trie.END in node


def test_item_description_without_blurb_matches_old_short_form():
    # about_the_game=None (the default) preserves the plain "Name — Genres"
    # form for callers that don't need the blurb (e.g. the popularity
    # baseline's SID-output tasks).
    assert item_description("Portal 2", "Puzzle,Comedy") == "Portal 2 — Puzzle, Comedy"


def test_item_description_appends_truncated_blurb_when_given():
    desc = item_description("Half-Life 2", "Action", "A first-person shooter set in City 17. More detail follows here.")
    assert desc == "Half-Life 2 — Action. A first-person shooter set in City 17."


def test_item_description_omits_separator_when_blurb_is_missing():
    assert item_description("Portal 2", "Puzzle,Comedy", None) == "Portal 2 — Puzzle, Comedy"
    assert item_description("Portal 2", "Puzzle,Comedy", "") == "Portal 2 — Puzzle, Comedy"


# ---------------------------------------------------------------------
# build_name_lookup
# ---------------------------------------------------------------------


def test_build_name_lookup_maps_full_description_to_plain_name(tiny_catalog):
    lookup = build_name_lookup(tiny_catalog)

    hl2_full_desc = item_description("Half-Life 2", "Action", tiny_catalog.row(0, named=True)["About the game"])
    portal2_full_desc = item_description("Portal 2", "Puzzle,Comedy", None)

    assert lookup[hl2_full_desc] == "Half-Life 2"
    assert lookup[portal2_full_desc] == "Portal 2"


def test_build_name_lookup_also_maps_the_short_form_for_asy(tiny_catalog):
    # asy's real target is the short "Name — Genres" form, distinct from
    # grounding_id2name's blurb-enriched target for items that have a
    # blurb (Half-Life 2 here) -- both must resolve to the same Name.
    lookup = build_name_lookup(tiny_catalog)
    hl2_short_desc = item_description("Half-Life 2", "Action")
    assert lookup[hl2_short_desc] == "Half-Life 2"


def test_build_name_lookup_collapses_duplicate_names_to_the_same_value(tiny_catalog):
    # Two different catalog rows sharing a Name (e.g. a re-released title)
    # must both map to that same Name -- the point of this lookup is to
    # score "predicted the correct game", and duplicate-named rows are
    # indistinguishable from the model's perspective once decoded. Checked
    # via the set of Name values (not raw key count), since each item can
    # legitimately contribute up to 2 distinct description-string keys
    # (short + long form) without that affecting how many distinct games
    # the lookup actually resolves to.
    duped = pl.concat([tiny_catalog, tiny_catalog.head(1)])
    lookup = build_name_lookup(duped)
    assert set(lookup.values()) == {"Half-Life 2", "Portal 2"}


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


# ---------------------------------------------------------------------
# build_sid_criteria_lookup / criteria_satisfied_at_k / criteria_ndcg_at_k
# ---------------------------------------------------------------------


def test_build_sid_criteria_lookup_splits_genres_and_categories(tiny_catalog):
    lookup = build_sid_criteria_lookup(tiny_catalog)
    sid = semantic_id_to_tokens([1, 2, 3, 0])  # Half-Life 2, Genres="Action"
    assert lookup[sid]["genres"] == {"Action"}
    assert lookup[sid]["categories"] == set()  # tiny_catalog fixture has no Categories column data


LOOKUP = {
    "sid_action_mp": {"genres": {"Action"}, "categories": {"Multi-player"}},
    "sid_action_only": {"genres": {"Action"}, "categories": set()},
    "sid_rpg": {"genres": {"RPG"}, "categories": {"Single-player"}},
}


def test_criteria_satisfied_at_k_hit_when_genre_matches():
    assert criteria_satisfied_at_k(["sid_action_only"], {"genres": ["Action"]}, LOOKUP, k=5) == 1.0


def test_criteria_satisfied_at_k_miss_when_genre_absent():
    assert criteria_satisfied_at_k(["sid_rpg"], {"genres": ["Action"]}, LOOKUP, k=5) == 0.0


def test_criteria_satisfied_at_k_requires_all_criteria():
    # sid_action_only has the genre but not the category -- must not count.
    assert criteria_satisfied_at_k(["sid_action_only"], {"genres": ["Action"], "categories": ["Multi-player"]}, LOOKUP, k=5) == 0.0
    assert criteria_satisfied_at_k(["sid_action_mp"], {"genres": ["Action"], "categories": ["Multi-player"]}, LOOKUP, k=5) == 1.0


def test_criteria_satisfied_at_k_ignores_unknown_candidate():
    """A candidate missing from the lookup (shouldn't happen under
    constrained decoding, but shouldn't crash the metric either)."""
    assert criteria_satisfied_at_k(["not_a_real_sid"], {"genres": ["Action"]}, LOOKUP, k=5) == 0.0


def test_criteria_satisfied_at_k_respects_k_boundary():
    candidates = ["sid_rpg", "sid_action_only"]  # match is at rank 2
    assert criteria_satisfied_at_k(candidates, {"genres": ["Action"]}, LOOKUP, k=1) == 0.0
    assert criteria_satisfied_at_k(candidates, {"genres": ["Action"]}, LOOKUP, k=2) == 1.0


def test_criteria_ndcg_at_k_rank_1_is_perfect_score():
    assert criteria_ndcg_at_k(["sid_action_only"], {"genres": ["Action"]}, LOOKUP, k=5) == pytest.approx(1.0)


def test_criteria_ndcg_at_k_decreases_with_rank():
    ndcg_rank_1 = criteria_ndcg_at_k(["sid_action_only", "sid_rpg"], {"genres": ["Action"]}, LOOKUP, k=5)
    ndcg_rank_2 = criteria_ndcg_at_k(["sid_rpg", "sid_action_only"], {"genres": ["Action"]}, LOOKUP, k=5)
    assert ndcg_rank_1 > ndcg_rank_2 > 0


def test_criteria_ndcg_at_k_zero_when_no_match():
    assert criteria_ndcg_at_k(["sid_rpg"], {"genres": ["Action"]}, LOOKUP, k=5) == 0.0
