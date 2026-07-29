# LLM-based Recommender System

A generative recommender pipeline: game catalog items are embedded, compressed into short
**semantic IDs** via an RQ-VAE, and those semantic IDs become a vocabulary an LLM (Qwen3) is
fine-tuned to reason over — predicting a user's next game, mapping IDs to/from game names, and
suggesting similar games — instead of the LLM ever seeing raw item IDs or embeddings directly.

## Pipeline

| # | Stage | Script | Output |
|---|---|---|---|
| 0 | Data prep | `notebooks/preprocess_australian_data.ipynb` (one-shot; not re-run) | `data/clean_game_catalog.parquet`, `data/clean_user_sequences.parquet` |
| 1 | Item embeddings | `src/build_game_embeddings.py` (orchestrates `tokenize_items.py` + `embed_items.py`, embeds with Qwen3-0.6B) | `data/output/games_with_embeddings.parquet` |
| 2 | RQ-VAE training | `src/train_rqvae.py` (config: `src/config.py:RQVAEConfig`) | `checkpoints/`, TensorBoard logs in `runs/` |
| 3 | Semantic ID export | `src/export_semantic_ids.py` (optionally appends a disambiguation digit for items that collide) | `data/output/semantic_ids.parquet` |
| 4 | Fine-tuning dataset | `src/build_finetune_dataset.py` (Alpaca-format SFT: sequential, grounding, similar-item, asy, nl_similar_item, nl_preference) | `data/output/sft_train.jsonl`, `sft_val.jsonl`, `sft_special_tokens.json` |
| 5 | Embedding warmup | `src/warmup_embeddings.py` (Stage 1 — trains only `embed_tokens`/`lm_head` on the new semantic-ID vocabulary) | adapter checkpoint under `outputs/qwen3-4b-embed-warmup/` |
| 6 | QLoRA fine-tune | `src/qlora_finetune.py` (Stage 2 — QLoRA over Stage 1's checkpoint via Unsloth) | adapter checkpoint under `outputs/qwen3-4b-qlora/` |
| 7 | Eval (recommendations) | `src/evaluate_ranking_metrics.py` (Recall@K/NDCG@K via constrained beam search) | per-task metrics printed to stdout |
| 7 | Eval (baseline) | `src/evaluate_popularity_baseline.py` ("always recommend the most popular item" sanity check) | per-task metrics printed to stdout |

Each numbered stage reads the previous stage's output, so run them in order the first time.
Stages 5+6 can also be run together via `python src/run_retrain.py`, which orchestrates both
with the Qwen3-4B / 4-bit overrides (the per-stage scripts default to the older Qwen3-0.6B /
non-quantized path).

## Tasks trained into the model

The SFT dataset mixes seven task types:

- **sequential** — predict the next item's semantic ID from a user's play history.
- **grounding_name2id / grounding_id2name** — map a semantic ID ↔ item name + genres
  (both directions). `grounding_id2name` is enriched with a short "About the game"
  snippet per STAR (arXiv 2604.02324), capped at 30 words.
- **similar_item** — given a semantic ID, recommend another item real users also engaged
  with (ground truth from co-occurrence in user sequences, not just genre overlap).
- **asy** (asymmetric item prediction, from LC-Rec, arXiv 2311.09049) — same
  `(history, target)` pairs as sequential, but the target is rendered as its
  name+genres text instead of its semantic ID. Reuses sequential's larger example pool
  to reinforce the index↔language link.
- **nl_similar_item** — same co-occurrence ground truth as `similar_item`, with the
  seed item rendered as a natural-language name reference ("recommend something like
  `<name>`") instead of a semantic ID.
- **nl_preference** — open-ended natural-language preference queries ("I want to play
  racing games", "action game with multiplayer") → a real matching item's semantic ID.
  Multiple valid targets per query (built from the catalog's Genres/Categories fields);
  evaluated on genre/category *consistency* rather than exact-match recall.

Three dataset-rebalancing fixes are baked into the build (see `build_finetune_dataset.py`'s
module docstring for the data behind them):

1. **Catalog restricted to interacted items** (~8.5k of the 93k-item catalog). The RQ-VAE
   codebook itself is reused as-is — 256 codes per level is far more room than 8.5k items
   need, so collisions don't increase, but exposure per item jumps ~11x.
2. **Floor/ceiling rebalancing per task.** Popularity-skewed targets are capped, and
   rare items are oversampled so the model can't collapse onto a few popular defaults.
3. **Cross-task exposure cap.** A single item can independently hit several tasks'
   ceilings at once; summed across tasks, popular items used to reach ~140 examples vs.
   a dozen for typical items. Capped at 40 per item (recommendation tasks only —
   grounding is excluded by design).

## Why two fine-tuning stages, and why QLoRA

The base model (Qwen3-4B) has never seen the semantic-ID tokens before — they're new vocabulary,
randomly initialized. Jumping straight into task-specific fine-tuning with random embeddings for
every ID makes the model spend most of its capacity just learning the token embeddings instead of
the actual tasks. Stage 1 warms up `embed_tokens`/`lm_head` alone (codebook-grounded
initialization + a short high-LR run) so the ID tokens already carry meaningful structure before
Stage 2 begins.

Both stages train through 4-bit quantization (QLoRA) because full-parameter fine-tuning of a ~4B
model needs far more VRAM than a single consumer GPU (12GB) provides for weights + gradients +
optimizer state. Stage 1 wraps a trivial rank-1 LoRA config purely to satisfy `peft`'s API — the
real training target there is `modules_to_save=["embed_tokens", "lm_head"]`. Stage 2 attaches a
real LoRA adapter (rank 8) across all attention/MLP projections on top of Stage 1's warmed-up
embeddings.

The Stage 1 codebook-grounded init is informed by STAR (arXiv 2604.02324, "Semantic-ID
Token-Embedding Alignment for Generative Recommenders") — see `src/warmup_embeddings.py`'s
module docstring for the four layered fixes (codebook-grounded init, full-sequence loss,
gradient-masked pretrained vocab, grounding-only sample) and the experiments that drove them.

## Constrained decoding & eval

`sid`-output tasks (sequential, similar_item, nl_similar_item, grounding_name2id, nl_preference)
are constrained at generation time to valid catalog semantic IDs via a prefix trie
(`src/constrained_decoding.py`), so every beam-search completion is guaranteed to decode to a
real catalog item. `grounding_id2name` is constrained to valid `"Name — Genres. <blurb>"`
descriptions. Ranking metrics (Recall@K, NDCG@K) are computed from beam-search candidates
ordered by beam score, the same way TIGER and LC-Rec evaluate generative recommenders.

`nl_preference` is scored with a *criteria-satisfaction* analog instead of exact-match recall
(many items can validly satisfy an open-ended query like "an action game").

A popularity baseline (`src/evaluate_popularity_baseline.py`) ranks the top-K globally popular
items, independent of input, as a sanity-check floor — beating random chance only shows the
model learned *something*; beating this baseline shows it's actually conditioning on the input.

## Project layout

- `src/` — pipeline code:
  - `build_game_embeddings.py`, `tokenize_items.py`, `embed_items.py` — Stage 1 (embeddings)
  - `config.py`, `rqvae.py`, `vector_quantizer.py`, `encoder.py`, `normalization.py`,
    `lr_scheduler.py`, `train_rqvae.py`, `export_semantic_ids.py` — Stage 2 (RQ-VAE)
  - `build_finetune_dataset.py` — Stage 4 (SFT dataset)
  - `warmup_embeddings.py`, `qlora_finetune.py`, `run_retrain.py` — Stages 5+6 (fine-tuning)
  - `constrained_decoding.py`, `evaluate_ranking_metrics.py`, `evaluate_popularity_baseline.py`
    — Stage 7 (eval)
  - `logger.py` — shared logging
- `data/` — raw + processed data (gitignored)
- `checkpoints/` — RQ-VAE checkpoints (gitignored)
- `runs/` — TensorBoard logs for RQ-VAE training (gitignored)
- `outputs/` — fine-tuning run checkpoints/experiments (gitignored — regenerate by re-running stages 5-6)
- `models/` — the final, stable QLoRA adapter (`models/qwen3-4b-qlora/`, gitignored — the weights are a 4.5GB binary, not meant for regular git history; see "Getting the model weights" below)
- `notebooks/`:
  - `preprocess_australian_data.ipynb` — one-shot data prep (Stage 0): cleans the Australian Steam Users dataset into the per-user sequences + item catalog the rest of the pipeline consumes.
  - `interactive_model_query.ipynb` — ask the fine-tuned model questions directly
  - `evaluate_semantic_ids.ipynb` — RQ-VAE codebook quality
- `tests/` — pytest unit tests for the core model pieces (vector quantizer, RQ-VAE, normalization,
  LR scheduler, embed-items, constrained-decoding, build-finetune-dataset). Run with
  `.venv/bin/python -m pytest -m "not slow"`. The single `slow`-marked test loads real model + GPU
  + data files; skip it for a fast run.
- `scratch_qualitative_eval.py` — one-off script: prints 10 sequential + 10 similar-item examples
  with the model's top-1 beam-search prediction vs. the expected item. Reads from
  `models/qwen3-4b-qlora-v2` (override `adapter_path` to point at your model).

## Setup

Requires Python 3.12 and a CUDA-capable NVIDIA GPU (12GB+ recommended) for the embedding/RQ-VAE/
fine-tuning stages. Dependencies are in `requirements.txt` — install in this order (see that
file's header comment for why unsloth is a separate final step):

```
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
pip install unsloth==2026.7.2 --no-deps
```

`unsloth` patches `transformers`/`trl`/`peft` at import time, so it must be imported before them
wherever it's used (already handled in `warmup_embeddings.py`/`qlora_finetune.py`).

## Running the pipeline

```
python src/build_game_embeddings.py    # Stage 1: tokenize + embed the catalog
python src/train_rqvae.py              # Stage 2: train the RQ-VAE
python src/export_semantic_ids.py      # Stage 3: encode items to semantic IDs
python src/build_finetune_dataset.py   # Stage 4: build the SFT dataset
python src/run_retrain.py              # Stages 5+6: warmup + QLoRA fine-tune (Qwen3-4B / 4-bit)
```

Or run the fine-tuning stages separately with explicit configs (useful if you want to tweak one
stage in isolation):

```
python src/warmup_embeddings.py        # Stage 5 — embedding warmup
python src/qlora_finetune.py           # Stage 6 — QLoRA fine-tune, loads Stage 1's checkpoint
```

Both stages default to conservative batch sizes (`micro_batch_size=1` with gradient accumulation)
tuned for a 12GB GPU — the training loop logs GPU memory periodically; watch for it sitting near
the card's ceiling (a Windows-specific failure mode falls back to slow shared system memory
instead of raising a clean OOM, silently slowing training 10-150x). Reduce the effective batch
size or `lora_r` if that happens. Both scripts save periodic checkpoints
(`save_steps`) rather than relying solely on the final save, since a crash or manual interruption
mid-run is recoverable from the last checkpoint via `resume_from_checkpoint`.

## Evaluating the fine-tuned model

```
python src/evaluate_ranking_metrics.py    # Recall@K / NDCG@K for every task (constrained beam search)
python src/evaluate_popularity_baseline.py # "always-recommend-popular" sanity-check floor
```

Both scripts sample `n=500` val examples per task by default; pass `--n` to change. The ranking
script also accepts `--temperature` (beam-search multinomial sampling) and `--source {val,train}`
(use `train` only as a diagnostic — it evaluates against examples the model already saw).

## Querying the fine-tuned model

`notebooks/interactive_model_query.ipynb` loads `models/qwen3-4b-qlora` and exposes convenience
functions per task (`ask_name2id`, `ask_id2name`, `ask_sequential`, `ask_asy`, `ask_similar`,
`round_trip`) for interactively probing what the model does and doesn't understand about the
semantic-ID vocabulary, with optional constrained decoding (guaranteed real catalog items) and
temperature sampling.

## Getting the model weights

`models/qwen3-4b-qlora/` isn't committed (4.5GB, no Git LFS set up in this repo). To reproduce it,
run the pipeline above end to end, then copy the final checkpoint's essential files out of
`outputs/qwen3-4b-qlora/checkpoint-<N>/` (`adapter_config.json`, `adapter_model.safetensors`,
`chat_template.jinja`, `tokenizer.json`, `tokenizer_config.json`, `README.md`) into `models/
qwen3-4b-qlora/` — that directory is a stable copy, decoupled from `outputs/`'s churn across
experiments. If you need to hand the model off without retraining, copy that directory directly.