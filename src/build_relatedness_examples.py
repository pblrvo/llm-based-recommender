"""Builds the semantic-ID relatedness task: given two item semantic IDs,
predict whether they belong to the same broad family (shared level-0 code).
"""

import random

RELATEDNESS_INSTRUCTIONS = [
    "Do these two games belong to the same broad family? Answer Yes or No.",
    "Are these two semantic IDs from the same game family?",
    "Determine whether these two items share the same broad category.",
    "Are Game A and Game B from the same broad game family? Answer Yes or No.",
    "Do these two semantic IDs belong to the same item family?",
    "Judge whether these two games are part of the same broad group.",
    "Is Game B in the same broad family as Game A? Answer Yes or No.",
    "Given these two games, say whether they share a broad family.",
]


def build_relatedness_examples(
    item_codes: dict,   # item_id -> tuple(level codes), e.g. (89, 210, 246, 0)
    item_tokens: dict,  # item_id -> full "<|sid_start|>...<|sid_end|>" string
    n_per_item: int,
    rng: random.Random,
) -> list:
    """One positive + one negative example per item, per n_per_item repeat."""
    by_l0 = {}
    for item_id, codes in item_codes.items():
        by_l0.setdefault(codes[0], []).append(item_id)

    all_items = list(item_codes.keys())
    examples = []

    for item_id, codes in item_codes.items():
        same_family = [i for i in by_l0[codes[0]] if i != item_id]
        if not same_family:
            continue

        for _ in range(n_per_item):
            partner = rng.choice(same_family)
            examples.append(_render(item_id, partner, item_codes, item_tokens, label=True, rng=rng))

            partner = rng.choice(all_items)
            while item_codes[partner][0] == codes[0]:
                partner = rng.choice(all_items)
            examples.append(_render(item_id, partner, item_codes, item_tokens, label=False, rng=rng))

    return examples


def _render(item_a, item_b, item_codes, item_tokens, label: bool, rng: random.Random) -> dict:
    a, b = item_tokens[item_a], item_tokens[item_b]
    if rng.random() < 0.5:
        a, b = b, a
    return {
        "instruction": rng.choice(RELATEDNESS_INSTRUCTIONS),
        "input": f"Game A: {a}\nGame B: {b}",
        "output": "Yes" if label else "No",
        "task": "relatedness",
        "_target": item_a,
        "_label": label,
    }
