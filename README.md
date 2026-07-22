# LLM-based Recommender System

A generative recommender pipeline: game catalog items are embedded, compressed into short
**semantic IDs** via an RQ-VAE, and those semantic IDs become a vocabulary an LLM (Qwen3) is
fine-tuned to reason over — predicting a user's next game, mapping IDs to/from game names, and
suggesting similar games — instead of the LLM ever seeing raw item IDs or embeddings directly.

## Pipeline

| Stage | Script | Output |
|---|---|---|
| 1. Item embeddings | `src/build_game_embeddings.py` (tokenizes catalog, embeds with Qwen3-0.6B) | `data/output/games_with_embeddings.parquet` |
| 2. RQ-VAE training | `src/train_rqvae.py` (config: `src/config.py:RQVAEConfig`) | `checkpoints/`, TensorBoard logs in `runs/` |
| 3. Semantic ID export | `src/export_semantic_ids.py` | `data/output/semantic_ids.parquet` |
| 4. Fine-tuning dataset | `src/build_finetune_dataset.py` (SFT examples: sequential recommendation, ID↔name grounding, similar-item, asymmetric/ASY prediction) | `data/output/sft_train.jsonl`, `sft_val.jsonl`, `sft_special_tokens.json` |
| 5. Embedding warmup | `src/warmup_embeddings.py` (Stage 1 — trains only `embed_tokens`/`lm_head` on the new semantic-ID vocabulary before task-specific fine-tuning) | adapter checkpoint under `outputs/` |
| 6. QLoRA fine-tune | `src/qlora_finetune.py` (Stage 2 — QLoRA over Stage 1's checkpoint via Unsloth) | adapter checkpoint under `outputs/qwen3-4b-qlora/` |

Each stage reads the previous stage's output, so run them in order the first time.

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

## Project layout

- `src/` — pipeline code
- `data/` — raw + processed data (gitignored)
- `checkpoints/` — RQ-VAE checkpoints (gitignored)
- `runs/` — TensorBoard logs for RQ-VAE training (gitignored)
- `outputs/` — fine-tuning run checkpoints/experiments (gitignored — regenerate by re-running stages 5-6)
- `models/` — the final, stable QLoRA adapter (`models/qwen3-4b-qlora/`, gitignored — the weights are a 4.5GB binary, not meant for regular git history; see "Getting the model weights" below)
- `notebooks/` — `interactive_model_query.ipynb` (ask the fine-tuned model questions directly) and `evaluate_semantic_ids.ipynb` (RQ-VAE codebook quality)
- `tests/`

## Setup

Requires Python 3.12 and a CUDA-capable NVIDIA GPU (12GB+ recommended) for the embedding/RQ-VAE/
fine-tuning stages. Dependencies are in `requirements.txt` -- install in this order (see that
file's header comment for why unsloth is a separate final step):

```
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
pip install unsloth==2026.7.2 --no-deps
```

`unsloth` patches `transformers`/`trl`/`peft` at import time, so it must be imported before them
wherever it's used (already handled in `warmup_embeddings.py`/`qlora_finetune.py`).

## Running the fine-tuning stages

```
python src/build_finetune_dataset.py   # build the SFT dataset (stage 4)
python src/warmup_embeddings.py        # Stage 1 — embedding warmup
python src/qlora_finetune.py           # Stage 2 — QLoRA fine-tune, loads Stage 1's checkpoint
```

Both stages default to conservative batch sizes (`micro_batch_size=1` with gradient accumulation)
tuned for a 12GB GPU — the training loop logs GPU memory periodically; watch for it sitting near
the card's ceiling (a Windows-specific failure mode falls back to slow shared system memory
instead of raising a clean OOM, silently slowing training 10-150x). Reduce the effective batch
size or `lora_r` if that happens. Both scripts save periodic checkpoints
(`save_steps`) rather than relying solely on the final save, since a crash or manual interruption
mid-run is recoverable from the last checkpoint via `resume_from_checkpoint`.

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
