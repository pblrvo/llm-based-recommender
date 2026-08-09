import random
import re

from build_relatedness_examples import build_relatedness_examples


def _fake_catalog(n=200, seed=0):
    rng = random.Random(seed)
    item_codes, item_tokens = {}, {}
    for i in range(n):
        codes = (rng.randint(0, 9), rng.randint(0, 9), rng.randint(0, 9), 0)
        item_codes[i] = codes
        item_tokens[i] = "<|sid_start|>" + "".join(f"<|sid_L{lvl}_{c}|>" for lvl, c in enumerate(codes)) + "<|sid_end|>"
    return item_codes, item_tokens


def _parse_l0(token_str):
    return int(re.search(r"sid_L0_(\d+)", token_str).group(1))


def test_examples_carry_a_target_for_group_splitting():
    item_codes, item_tokens = _fake_catalog()
    examples = build_relatedness_examples(item_codes, item_tokens, n_per_item=3, rng=random.Random(1))
    assert examples
    assert all(ex["_target"] in item_codes for ex in examples)


def test_labels_match_actual_l0_sharing():
    item_codes, item_tokens = _fake_catalog()
    examples = build_relatedness_examples(item_codes, item_tokens, n_per_item=3, rng=random.Random(1))
    assert examples

    for ex in examples:
        game_a, game_b = ex["input"].split("\n")
        l0_a = _parse_l0(game_a)
        l0_b = _parse_l0(game_b)
        actually_same = l0_a == l0_b
        assert ex["output"] == ("Yes" if actually_same else "No")
        assert ex["_label"] == actually_same


def test_roughly_balanced_classes():
    item_codes, item_tokens = _fake_catalog()
    examples = build_relatedness_examples(item_codes, item_tokens, n_per_item=3, rng=random.Random(1))
    pos = sum(1 for e in examples if e["_label"])
    neg = len(examples) - pos
    assert pos == neg  # one positive + one negative per item per repeat, by construction


def test_negatives_sometimes_share_a_deeper_level():
    """Negatives must not always differ everywhere, or the model could learn
    'any shared token = related' instead of keying on L0 specifically."""
    item_codes, item_tokens = _fake_catalog(n=500, seed=2)
    examples = build_relatedness_examples(item_codes, item_tokens, n_per_item=5, rng=random.Random(2))

    negatives = [e for e in examples if not e["_label"]]
    share_deeper = 0
    for ex in negatives:
        game_a, game_b = ex["input"].split("\n")
        codes_a = tuple(int(c) for c in re.findall(r"sid_L\d+_(\d+)", game_a))
        codes_b = tuple(int(c) for c in re.findall(r"sid_L\d+_(\d+)", game_b))
        if any(a == b for a, b in zip(codes_a[1:], codes_b[1:])):
            share_deeper += 1

    assert share_deeper > 0, "negatives never share a deeper level -- risk of a trivial shortcut"
