# Fine-tuning dataset: changes and rationale

This document summarizes a round of data-quality and data-design work applied to the
SFT (instruction fine-tuning) dataset built by `src/build_finetune_dataset.py`, ahead
of a new training run. It compares the previous dataset (used to train the published
`Qwen3-4B-Game-semantic-IDs-v2` model) against the dataset produced after these
changes, and documents the reasoning and measured evidence behind each change so it
can be cited directly in the accompanying paper.

## 1. Summary

| | Previous dataset (v2) | New dataset (this work) |
|---|---|---|
| Total examples | 316,000 (per model card) | ~396,400 (371,182 train / 25,202 val) |
| Tasks | 7 | 8 (adds semantic-ID relatedness) |
| Sequence ground truth | Playtime-ordered, **unfiltered** — ~24.8% of a typical user's owned items have zero recorded playtime | Zero-playtime items filtered out before any task consumes a sequence |
| `similar_item` ranking | Raw co-occurrence count (popularity-biased) | Pointwise Mutual Information (PMI) |
| Instruction phrasing | 2–5 fixed templates per task | 6–12 templates per task |
| `nl_preference` volume | 20 examples/genre (of up to 5,399 qualifying items) | 150 examples/genre — real distinct targets, not repeats |
| Sequential exposure | Real user data only; rare items reach the floor via literal clones | Real data + synthetic k-NN-walk sequences for under-exposed items (train-only) |
| Grounding share of dataset | ~56% (dominant, crowding out relational tasks) | ~42% |

The rest of this document explains each row.

## 2. Ground-truth quality: playtime filtering

**Problem.** The Stage 0 preprocessing notebook (`notebooks/preprocess_australian_data.ipynb`)
orders each user's item sequence by `playtime_forever`, descending, as a proxy for
engagement (the source dataset has no interaction timestamps). However, the notebook's
own data-quality analysis shows **36.2% of all owned items have zero recorded
playtime** (bundles, free weekends, gifting — owned but never opened). The previous
pipeline discarded playtime after sorting, keeping only the ordered item IDs. This
meant:

- `sequential`/`asy` training pairs could have a **never-played game as the prediction
  target**, in a block whose internal order is arbitrary (all ties at playtime = 0).
- `similar_item`'s co-occurrence computation counted "both owned by the same user" as
  a similarity signal, without requiring either to have actually been played.

**Fix.** `notebooks/preprocess_australian_data.ipynb` was re-run against the raw
source data with an added `playtime_sequence` column, retaining each item's playtime
value alongside its ID (previously discarded). A new helper,
`AlpacaDatasetBuilder._played_sequence`, filters every sequence to non-zero-playtime,
catalog-known items before it is consumed by `_build_history_target_pairs` (feeds
`sequential`/`asy`) or `_compute_similar_partners` (feeds `similar_item`/
`nl_similar_item`). Measured on the regenerated data: the mean zero-playtime fraction
per user is 24.8%; median sequence length after filtering is 27 played items (p10 = 5,
p90 = 95).

## 3. `similar_item` ranking: PMI instead of raw co-occurrence count

**Problem.** Two items that are each independently very popular co-occur often simply
because most users own both — raw co-occurrence count conflates *mutual popularity*
with *genuine affinity*. Measured directly on the previous dataset: pairs selected as
"similar" by raw count shared a semantic-ID level-0 codebook prefix (a proxy for real
content similarity, see §4) only **6.8%** of the time, barely above a **1.1%** random
baseline within the same item pool (~6.2x enrichment).

**Fix.** `_compute_similar_partners` now ranks each item's co-occurring partners by
**Pointwise Mutual Information**, `PMI(a, b) = log(count(a, b) · N / (freq(a) · freq(b)))`,
where `N` is the number of qualifying co-occurrence windows and `freq` is each item's
window-frequency. A pair only ranks highly when it co-occurs *more than* the items'
individual popularity would predict by chance. The `min_cooccurrence` floor (2) is kept
to avoid PMI's known instability on singleton counts.

**Measured effect** (combined with the playtime fix in §2): level-0 prefix-sharing rose
from 6.8% to **10.8%**, against a 1.1% random baseline — enrichment improved from
~6.2x to **~9.5x**.

## 4. New task: semantic-ID relatedness

**Motivation.** Item semantic IDs are produced by an RQ-VAE with a hierarchical
codebook (coarse-to-fine levels). None of the 7 original tasks ever test whether the
model understands this internal structure — every task treats a semantic ID as one
opaque atomic block, either as input or output. Two properties of the ID space were
verified empirically before designing this task:

- **Level-0 codes do *not* cleanly encode genre.** Across 187 level-0 clusters
  (~490 items each, full catalog), the average top-genre purity is 31%, barely above
  the corpus-wide baseline (Indie alone is 24.5% of all genre tags). This ruled out an
  earlier candidate task ("what genre does this ID prefix represent?") — the labels
  would have been mostly noise.
- **Embedding-space neighbors *do* reliably share a level-0 code.** Sampling 2,000
  items, their top-10 nearest neighbors (by item-embedding cosine similarity) share the
  seed item's level-0 code 51.6% of the time, versus a 0.6% random baseline (~90x
  enrichment) — consistent with the RQ-VAE's nearest-centroid quantization at that
  level.

**Design.** `src/build_relatedness_examples.py` builds a binary classification task:
given two items' semantic IDs, predict whether they share the same broad family
(ground truth: shared level-0 code — computed directly from the codebook, not an
invented label). To prevent the model from learning a "any shared token = related"
shortcut, negative pairs deliberately include cases that still share a *deeper* level
(L1/L2/L3) while differing at L0, forcing the signal to key on the level-0 position
specifically.

**Volume.** 51,366 raw examples (3 positive + 3 negative per item), split by group
like the other relational tasks; excluded from the cross-task exposure cap since its
ground truth is structural, not popularity-shaped (same treatment as grounding).

## 5. Synthetic sequential data (k-NN embedding walks)

**Motivation.** After the fixes in §2–3, 2,985 of 8,563 items still fell under
`sequential`'s target-exposure floor from real data alone. The existing rebalancing
mechanism (`_rebalance_pairs_by_target`) would otherwise top these up with **literal
duplicate clones** of existing examples — low gradient-signal value.

**Method.** `src/build_synthetic_sequences.py` builds a k-NN graph over item
embeddings (cosine similarity, restricted to the interacted-item catalog) and
generates temperature-weighted random walks ending at each under-exposed item, with
walk length sampled from the real played-sequence length distribution. Because
embedding-space neighbors reliably share a level-0 codebook prefix (§4), these walks
implicitly reinforce prefix-sharing = meaning-sharing for items real data barely
covers — the same structural signal the relatedness task teaches explicitly.

**Guarantees enforced in the pipeline:**
- Synthetic rows are **excluded from `similar_item`'s co-occurrence computation** —
  including them would be circular, since the walks are themselves derived from
  embedding similarity, and would launder manufactured data as evidence of real
  affinity.
- Synthetic rows are tagged (`is_synthetic`) end-to-end and **actively migrated out of
  the validation split** post-hoc (`_exclude_synthetic_from_val`), so evaluation only
  ever measures performance against real user behavior.

**Volume.** 10,868 synthetic sequences generated; `sequential` train examples grew
from ~32,000 (real-only) to ~59,600 (real + synthetic) — largely replacing
low-information clone-oversampling with structurally-grounded diverse examples.

## 6. Instruction and query phrasing diversity

Every task's instruction pool (and the natural-language query templates for
`nl_preference`/`nl_similar_item`) was expanded from 2–5 fixed phrasings to 6–12,
reducing the risk of the model overfitting to surface phrasing rather than the
underlying task, and giving `grounding`'s train/val split (which holds out unseen
*phrasing* rather than unseen items — see §8) a meaningfully larger held-out space.

## 7. `nl_preference` volume

**Problem.** `nl_preference` queries are answered by sampling real catalog items
matching a genre/category — every example is a genuinely distinct (query, target)
pair, not a repeat, so there was no quality cost to showing more of them. Yet the
previous configuration capped this at 20 examples per genre, regardless of how many
qualifying items existed (up to 5,399 for "Indie").

**Fix.** Raised to 150 examples/genre and 40/combo — still under the smallest
qualifying genre's pool (212 items, "Massively Multiplayer"), so no genre needs
repeats to reach the new cap. Raw `nl_preference` examples grew from 1,330 to 6,055
(4.5x) at no quality cost.

## 8. Final task mix

| Task | Train | Val | Share of train |
|---|--:|--:|--:|
| `grounding_id2name` | 77,067 | 8,563 | 20.8% |
| `grounding_name2id` | 77,067 | 8,563 | 20.8% |
| `sequential` | 59,651 | 1,518 | 16.1% |
| `asy` | 59,449 | 1,517 | 16.0% |
| `relatedness` | 48,798 | 2,568 | 13.1% |
| `nl_similar_item` | 22,178 | 1,113 | 6.0% |
| `similar_item` | 22,070 | 1,142 | 5.9% |
| `nl_preference` | 4,902 | 218 | 1.3% |
| **Total** | **371,182** | **25,202** | 100% |

Grounding (`id2name` + `name2id`) still accounts for the largest single share
(41.6% combined) but dropped from an estimated ~56% of the previous dataset, giving
the relational/recommendation tasks comparatively more room.

**Note on exact reproducibility.** Re-running the pipeline with the same seed produces
counts within ~0.1% of the table above (not bit-identical): `_rebalance_by_target` and
similar sampling steps draw from a single shared `random.Random` instance whose call
sequence depends on dict/row iteration order, and polars' hash-join (`load_data`'s
`sid_df.join(catalog_df, ...)`) does not guarantee a stable row order across runs.
Task proportions and the qualitative conclusions in this document are unaffected.

**Deliberate scale decision.** The 8,563-item interacted catalog bounds how much
non-redundant signal exists per task. Reaching ~1M total examples was evaluated and
rejected: it would require either lowering `similar_item`'s co-occurrence floor from 2
to 1 (reintroducing the PMI-reliability problem from §3) or substantially raising
`sequential`'s exposure ceiling without a proportional increase in synthetic coverage
for the long tail (reintroducing the popularity-skew problem the floor/ceiling
rebalancing exists to prevent). ~400K was kept as the quality-calibrated target
instead of trading back the fixes above for raw volume.

## 9. Reproducibility

All changes are covered by unit tests (`tests/test_build_finetune_dataset.py`,
`tests/test_build_relatedness_examples.py`, `tests/test_build_synthetic_sequences.py`)
— 156 tests total. Pipeline order to reproduce the new dataset from raw data:

```
notebooks/preprocess_australian_data.ipynb   # Stage 0 (retains playtime_sequence)
python src/build_synthetic_sequences.py      # writes data/combined_user_sequences.parquet
python src/build_finetune_dataset.py --sequences-path data/combined_user_sequences.parquet
```
