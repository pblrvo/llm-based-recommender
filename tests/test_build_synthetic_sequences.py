import numpy as np

from build_synthetic_sequences import walk_to_target, build_knn, played_sequence, load_real_frequencies


def test_walk_ends_at_target_and_has_requested_length():
    rng = np.random.default_rng(0)
    embeddings = rng.normal(size=(20, 8)).astype(np.float32)
    nn = build_knn(embeddings, k=5)
    sim, idx = nn.kneighbors(embeddings)
    sim = 1.0 - sim

    for target_idx in (0, 5, 19):
        walk = walk_to_target(target_idx, length=6, neighbor_idx=idx, neighbor_sim=sim,
                               temperature=0.1, rng=rng)
        assert len(walk) == 6
        assert walk[-1] == target_idx
        assert all(0 <= i < 20 for i in walk)


def test_walk_length_one_is_just_the_target():
    rng = np.random.default_rng(0)
    embeddings = rng.normal(size=(10, 4)).astype(np.float32)
    nn = build_knn(embeddings, k=3)
    sim, idx = nn.kneighbors(embeddings)
    sim = 1.0 - sim

    walk = walk_to_target(3, length=1, neighbor_idx=idx, neighbor_sim=sim, temperature=0.1, rng=rng)
    assert walk == [3]


def test_played_sequence_drops_zero_playtime_and_unknown_items():
    row = {"item_sequence": ["x", "y", "z"], "playtime_sequence": [300, 0, 100]}
    assert played_sequence(row, interacted_ids={"x", "y", "z"}) == ["x", "z"]
    assert played_sequence(row, interacted_ids={"x"}) == ["x"]


def test_load_real_frequencies_ignores_zero_playtime_targets():
    import polars as pl

    sequences_df = pl.DataFrame({
        "item_sequence": [["x", "y", "z"]],
        "playtime_sequence": [[300, 0, 100]],  # y never played
        "is_long_tail_user": [False],
    })
    freq = load_real_frequencies(sequences_df, interacted_ids={"x", "y", "z"})
    assert freq == {"z": 1}  # only "z" is a played, non-first-position item
