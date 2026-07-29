"""Stage 1 of a two-stage fine-tuning strategy: warm up the new semantic-ID
token embeddings before any task-specific (LoRA) training happens.

Why this exists: a prior single-stage run (add tokens, then immediately do
task-specific LoRA training at a normal learning rate) converged on loss but
never learned the actual semantic-ID content -- 0% exact-match on grounding
tasks despite a healthy-looking eval_loss. The new tokens had to learn "what
am I" and "how do I get used for this task" simultaneously, which is a much
harder optimization problem than solving them separately.

This stage freezes every parameter except embed_tokens/lm_head and trains
only those, at a high learning rate, on a data sample -- giving the new
tokens a differentiated starting point before Stage 2 (this project's
existing axolotl LoRA pipeline, see finetune_qwen.py) does the real
task-specific training starting from this checkpoint instead of the raw
base model. Approach adapted from github.com/eugeneyan/semantic-ids-llm's
finetune_qwen3_8b_vocab.py.

Two further fixes on top of that, after a full two-stage run (embedding
warmup + rebalanced-data LoRA, then again with full fine-tuning/bigger
batch/more epochs) still produced 0% grounding exact-match:

1. Codebook-grounded initialization (see `_codebook_grounded_vectors`).
   HF's default `resize_token_embeddings` (`mean_resizing=True`) draws every
   new token from the *same* mean/covariance-based random distribution --
   nothing differentiates `<|sid_L0_87|>` from `<|sid_L2_200|>` at init. Per
   "Grounded Token Initialization for New Vocabulary in LMs for Generative
   Recommendation" (arXiv 2604.02324), this collapses new tokens into a
   degenerate subspace that fine-tuning struggles to fully recover
   inter-token distinctions from. We have real trained embeddings for these
   tokens already -- the RQ-VAE codebooks (checkpoints/rqvae_best.pt) -- so
   there's no need to let the LLM tokens start from nothing.
2. Full-sequence loss (`completion_only_loss=False`), matching eugeneyan's
   recipe: sid tokens get gradient signal every time they appear, including
   on the prompt side (history sequences, grounding_id2name's input ID),
   not only when the model is asked to generate them.

Two further fixes, informed by STAR (arXiv, "Semantic-ID Token-Embedding
Alignment for Generative Recommenders" -- an ICLR 2026 submission that
independently formalizes almost this exact stage, published after this
project's own codebook-grounded init was already in place):

3. Gradient-masking the pretrained vocabulary (see
   `_freeze_pretrained_vocab_gradient`). Making embed_tokens/lm_head
   trainable, as described above, makes the WHOLE embedding matrix
   trainable -- not just the 1026 new sid rows. Under full-sequence loss,
   every ordinary vocabulary token appearing in a prompt gets a real
   gradient too, so this stage's unusually high LR (1e-3) was quietly
   perturbing the pretrained vocabulary's embeddings alongside the new
   tokens. STAR's whole method is built on the opposite: freeze the
   pretrained vocabulary entirely, update only the new tokens' rows. A
   backward hook enforces the same guarantee here without giving up
   modules_to_save/full-matrix-requires_grad (both still needed for PEFT's
   API and tied-embedding bookkeeping) -- see also why weight_decay
   defaults to 0.0 now, since decoupled AdamW decay isn't gated by the
   gradient and would otherwise still shrink the "frozen" rows.
4. Grounding-only training sample (`EmbeddingWarmupConfig.grounding_only`).
   STAR's ablation shows their alignment gains come specifically from
   title/description<->Semantic-ID supervision, not from broader task
   exposure -- so this stage's sample is restricted to grounding_id2name/
   grounding_name2id by default instead of a mix of all 7 tasks, which
   only diluted that one signal. Stage 2's full SFT run still covers every
   task normally.

Plain transformers/TRL by default (`load_in_4bit=False`) -- this step is
short (~750-2250 steps) and doesn't need Unsloth's speed optimizations, and
this project hit a tied-embeddings/target_modules pitfall with Unsloth's
PEFT integration once before (see finetune_qwen.py's git history).

For models too large to hold frozen in bf16 alongside trainable embeddings
(e.g. Qwen3-4B, ~16GB unquantized vs. this project's 12GB GPU),
`load_in_4bit=True` switches to a quantized backbone via Unsloth's
FastLanguageModel instead of plain transformers + bitsandbytes + peft
directly -- the latter combination hit an unresolved dtype bug (float vs
bfloat16 at the lm_head layer) that only reproduced inside the real
SFTTrainer, not in any isolated test of the same pieces, pointing at a
Trainer/accelerate internals interaction rather than anything fixable in
this file. Unsloth maintains tested patches for exactly this scenario
(quantized backbone + newly added trainable vocab) -- the same combination
eugeneyan's reference project used successfully for this same purpose.
"""

from unsloth import FastLanguageModel, add_new_tokens  # isort: skip -- must import before transformers/trl/peft, see below

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import SFTConfig, SFTTrainer

from logger import Logger

logger = Logger.get_logger(__name__)

# Number of RQ-VAE levels with a real trained codebook. The sid token scheme
# has a 4th level (<|sid_L3_*|>) but it's a collision-disambiguation digit
# added in build_finetune_dataset.py, not a learned RQ-VAE level -- almost
# always 0, no codebook vector exists for it.
CODEBOOK_LEVELS = 3

# Task names the EmbeddingWarmupConfig.grounding_only sample restricts to --
# the title/description<->Semantic-ID pairs, matching STAR's alignment-stage
# corpus (see _freeze_pretrained_vocab_gradient's docstring for the paper).
GROUNDING_TASKS = {"grounding_id2name", "grounding_name2id"}


def _codebook_grounded_vectors(
    rqvae_checkpoint_path: Path, hidden_size: int, target_norm: float, generator: torch.Generator,
) -> Dict[str, torch.Tensor]:
    """Build init vectors for the L0-L2 sid tokens from the trained RQ-VAE codebooks.

    Projects each level's 32-dim codebook into the LLM embedding space with
    a fixed isometric projection (orthonormal columns from QR-decomposing a
    seeded random Gaussian), then rescales per level so norms land where
    the rest of the vocabulary already lives.

    The isometry (not just an approximately distance-preserving random
    projection) is deliberate: with orthonormal columns Q,
    ||Q v1 - Q v2|| == ||v1 - v2|| *exactly*, so the RQ-VAE codebook's
    relative geometry -- which codes are close/far apart, and level 0's
    meaningfully larger norm variation vs. levels 1/2's near-uniform
    norms -- transfers into the LLM's embedding space unchanged, just
    uniformly rescaled per level.
    """
    state_dict = torch.load(rqvae_checkpoint_path, map_location="cpu", weights_only=False)["model_state_dict"]

    vectors: Dict[str, torch.Tensor] = {}
    for level in range(CODEBOOK_LEVELS):
        codebook = state_dict[f"vq_layers.{level}.embedding.weight"].float()  # (256, code_dim)
        num_codes, code_dim = codebook.shape

        projection, _ = torch.linalg.qr(torch.randn(hidden_size, code_dim, generator=generator))
        projected = codebook @ projection.T  # (256, hidden_size), exact isometry

        scale = target_norm / codebook.norm(dim=-1).mean()
        projected = projected * scale

        for code in range(num_codes):
            vectors[f"<|sid_L{level}_{code}|>"] = projected[code]

        logger.info(
            "Level %d codebook: raw norm mean=%.4f -> projected norm mean=%.4f (target=%.4f)",
            level, codebook.norm(dim=-1).mean().item(), projected.norm(dim=-1).mean().item(), target_norm,
        )

    return vectors


def _distinct_random_vectors(
    tokens: List[str], hidden_size: int, target_norm: float, generator: torch.Generator,
) -> Dict[str, torch.Tensor]:
    """Build init vectors for sid tokens with no real learned codebook to ground them in.

    Used for sid_start, sid_end, and the L3 collision-disambiguation digit.
    Independent random unit vectors, scaled to target_norm.

    Exists because HF's default `resize_token_embeddings` turned out to be
    worse than this simple fallback, both theoretically (per the GTI paper)
    and empirically here: a smoke test measured sid_start and sid_end --
    left at the default init -- landing at cosine similarity 0.9999997, i.e.
    functionally the same vector, before this function existed. Independent
    draws in a 1024-dim space are nearly orthogonal to each other with high
    probability, which is what actually differentiates tokens.
    """
    raw = torch.randn(len(tokens), hidden_size, generator=generator)
    unit = raw / raw.norm(dim=-1, keepdim=True)
    scaled = unit * target_norm
    return {token: scaled[i] for i, token in enumerate(tokens)}


def _sid_token_init_vectors(
    rqvae_checkpoint_path: Path, hidden_size: int, target_norm: float, seed: int,
) -> Dict[str, torch.Tensor]:
    """Build init vectors for all 1026 new sid tokens: codebook-grounded for L0-L2, distinct random for the rest."""
    generator = torch.Generator().manual_seed(seed)

    vectors = _codebook_grounded_vectors(rqvae_checkpoint_path, hidden_size, target_norm, generator)

    other_tokens = ["<|sid_start|>", "<|sid_end|>"] + [f"<|sid_L3_{code}|>" for code in range(256)]
    vectors.update(_distinct_random_vectors(other_tokens, hidden_size, target_norm, generator))

    return vectors


def _freeze_pretrained_vocab_gradient(model, original_vocab_size: int) -> None:
    """Register a backward hook that zeros gradients on pretrained vocab rows.

    Both the input and (if untied) output embedding weights get a hook that
    zeroes rows < original_vocab_size. Only the newly added sid tokens
    (rows >= original_vocab_size) ever receive a nonzero gradient.

    Zeroing the gradient (rather than pursuing a real "only these rows are
    Parameters" split, which isn't practical here given the frozen backbone
    already needs the full matrix wrapped via modules_to_save/PEFT for
    other reasons) also means AdamW's decoupled weight decay would still
    silently shrink the "frozen" rows every step, since decoupled decay is
    applied directly to the parameter and doesn't look at the gradient at
    all -- that's why EmbeddingWarmupConfig.weight_decay defaults to 0.0.
    """
    seen = set()
    for get_embeddings in (model.get_input_embeddings, model.get_output_embeddings):
        weight = get_embeddings().weight
        if id(weight) in seen:
            continue  # tied embeddings -- same tensor, one hook is enough
        seen.add(id(weight))

        def _mask_pretrained_rows(grad, original_vocab_size=original_vocab_size):
            grad[:original_vocab_size] = 0
            return grad

        weight.register_hook(_mask_pretrained_rows)
    logger.info(
        "Registered gradient mask: only embedding rows >= %d (the %d new sid tokens) will update",
        original_vocab_size, weight.shape[0] - original_vocab_size,
    )


ALPACA_PROMPT = (
    "Below is an instruction that describes a task, paired with an input that provides further context. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
)

# Fixed probes for the live generation check, one per task type, covering
# both directions of grounding since that's what completely failed before.
GENERATION_PROBES = [
    (
        "What is the semantic ID for this game?",
        "Half-Life 2",
    ),
    (
        "A player enjoyed this game. Suggest another game they would likely also enjoy.",
        None,  # filled in at runtime with a real semantic ID from the dataset
    ),
    (
        "Given a user's game history, ordered from most to least played, predict the name of the next game they are likely to enjoy.",
        None,  # filled in at runtime with a real multi-item history from the dataset
    ),
]


@dataclass
class EmbeddingWarmupConfig:
    """Configuration for the Stage 1 embedding-warmup run."""

    data_dir: Path = Path("data")
    train_path: Optional[Path] = None
    special_tokens_path: Optional[Path] = None
    output_dir: Path = Path("outputs/qwen3-0.6b-embed-warmup")
    rqvae_checkpoint_path: Path = Path("checkpoints/rqvae_best.pt")

    base_model: str = "Qwen/Qwen3-0.6B"
    # 4B (and any model too big to comfortably hold frozen weights in bf16
    # alongside trainable embeddings) can load the frozen backbone in 4-bit
    # -- embed_tokens/lm_head stay unquantized/trainable either way, since
    # only Linear layers get bitsandbytes-wrapped, not the embedding table.
    # lm_head is explicitly excluded too (llm_int8_skip_modules) since,
    # despite being tied to embed_tokens, it's still its own nn.Linear that
    # would otherwise get quantized separately and break the tie.
    load_in_4bit: bool = False
    max_seq_length: int = 192

    # Only a sample -- this stage is about giving the new tokens a sane
    # starting point, not full convergence (that's Stage 2's job).
    max_training_samples: int = 20000
    # Restricts the sample to grounding_id2name/grounding_name2id (title<->
    # sid pairs) instead of all 7 task types. Matches STAR's (arXiv,
    # "Semantic-ID Token-Embedding Alignment for Generative Recommenders")
    # alignment-stage design directly -- their ablation shows the gain comes
    # specifically from item title/description<->Semantic-ID supervision,
    # not from broader task exposure, so mixing in sequential/similar_item/
    # nl_preference examples here dilutes the one signal this stage actually
    # needs. Stage 2's full SFT run still covers every task normally.
    grounding_only: bool = True

    # High LR relative to normal fine-tuning (Stage 2 uses 2e-4): the only
    # trainable parameters are the two embedding matrices, freshly
    # initialized for 1026 tokens, so they need to move fast.
    learning_rate: float = 1e-3
    # Was 16 with grad_accum=1. Full-sequence loss (completion_only_loss=
    # False) computes logits/loss over every position, not just the
    # completion -- unlike completion-only loss, this can't skip masked-out
    # positions, so it costs meaningfully more memory. Measured: batch=16
    # alone hit ~5.5GB allocated / ~7.65GB reserved on step 1 of this 12GB
    # GPU, close enough to the ceiling that Windows' silent GPU-memory
    # "shared" fallback kicked in and turned ~1.3s steps into 20-150s+
    # steps. Smaller micro-batch + gradient accumulation keeps the same
    # effective batch size (16) at a fraction of the peak memory.
    micro_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_steps: int = 750
    warmup_steps: int = 50
    # Was 0.01. Decoupled AdamW weight decay is applied to every parameter
    # directly, independent of its gradient -- so even with the pretrained
    # vocabulary's rows gradient-masked to zero (see
    # _freeze_pretrained_vocab_gradient), nonzero weight decay would still
    # shrink them a little every step. Nothing in this stage should touch
    # the pretrained vocabulary at all, so decay is off entirely rather
    # than relying on the (tiny) effect being negligible.
    weight_decay: float = 0.0
    lr_scheduler_type: str = "cosine"
    optimizer: str = "adamw_torch"
    logging_steps: int = 10
    generation_check_steps: int = 150
    seed: int = 0

    # None (default) keeps the original behavior: no intermediate
    # checkpoints, only the final save_pretrained() call. Set both to
    # inspect progress at multiple points along a longer run instead of
    # only the final state.
    save_steps: Optional[int] = None
    save_total_limit: int = 3

    def __post_init__(self):
        """Fill in computed defaults and validate the checkpoint/quantization combination."""
        if self.train_path is None:
            self.train_path = self.data_dir / "output" / "sft_train.jsonl"
        if self.special_tokens_path is None:
            self.special_tokens_path = self.data_dir / "output" / "sft_special_tokens.json"

        if self.load_in_4bit and self.save_steps is None:
            # The quantized path's final save is a known-broken no-op (see
            # train()) -- periodic checkpointing is the ONLY way a run's
            # result survives. Without this, a completed run would leave
            # nothing to recover.
            raise ValueError(
                "load_in_4bit=True requires save_steps to be set -- the "
                "final save is skipped for quantized runs (see train()'s "
                "docstring), so periodic checkpoints are the only output."
            )

        logger.info(
            "EmbeddingWarmupConfig: base_model=%s, max_training_samples=%d, lr=%.2e, "
            "max_steps=%d, micro_batch_size=%d",
            self.base_model, self.max_training_samples, self.learning_rate,
            self.max_steps, self.micro_batch_size,
        )


class GenerationCheckCallback(TrainerCallback):
    """Run a couple of fixed probes through the model periodically so problems surface within minutes."""

    def __init__(self, tokenizer, probes: List[tuple], interval: int):
        """Store the tokenizer, fixed probes, and step interval."""
        self.tokenizer = tokenizer
        self.probes = probes
        self.interval = interval

    def _run(self, model, step: int):
        """Run every probe through `model` in eval mode and log the outputs."""
        was_training = model.training
        model.eval()
        logger.info("=== Generation check at step %d ===", step)
        with torch.no_grad():
            for instruction, user_input in self.probes:
                prompt = ALPACA_PROMPT.format(instruction=instruction, input=user_input)
                inputs = self.tokenizer(prompt, return_tensors="pt").to(model.device)
                output_ids = model.generate(
                    **inputs, max_new_tokens=24, do_sample=False,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                )
                new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
                decoded = self.tokenizer.decode(new_tokens, skip_special_tokens=False)
                logger.info("  input=%r -> output=%r", user_input, decoded)
        model.train(was_training)

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """Run probes at step 0 (before training begins)."""
        self._run(model, 0)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Run probes at every Nth training step."""
        if state.global_step > 0 and state.global_step % self.interval == 0:
            self._run(model, state.global_step)


class EmbeddingWarmupTrainer:
    """Stage 1 trainer: load the base LM with sid tokens added, train embed_tokens/lm_head only."""

    def __init__(self, config: EmbeddingWarmupConfig):
        """Store the config; load the model and dataset on demand."""
        self.config = config
        self.model = None
        self.tokenizer = None

    def _load_special_tokens(self) -> List[str]:
        """Read the sid special tokens from disk."""
        path = self.config.special_tokens_path
        if not path.exists():
            raise FileNotFoundError(f"Special tokens file not found at {path}. Run build_finetune_dataset.py first.")
        with open(path, encoding="utf-8") as f:
            tokens = json.load(f)
        logger.info("Loaded %d special tokens from %s", len(tokens), path)
        return tokens

    def load_model(self):
        """Load base model + tokenizer, add sid tokens, init their weights, and make them trainable.

        Returns:
            (model, tokenizer) pair ready for SFTTrainer.
        """
        cfg = self.config
        special_tokens = self._load_special_tokens()

        if cfg.load_in_4bit:
            # Plain transformers + bitsandbytes + peft hit an unresolved
            # dtype bug here: every individual piece (quantized loading,
            # resize, custom init, PEFT wrapping, even generate() itself)
            # worked in isolation, but failed once run through the real
            # SFTTrainer -- pointing at a Trainer/accelerate internals
            # interaction, not our code. Unsloth maintains hand-tested
            # patches for exactly this scenario (quantized backbone + newly
            # added trainable vocab), and it's the same combination
            # eugeneyan's reference project used successfully for the same
            # Stage 1 purpose.
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=cfg.base_model,
                max_seq_length=cfg.max_seq_length,
                dtype=torch.bfloat16,
                load_in_4bit=True,
            )

            existing_norm_mean = model.get_input_embeddings().weight.norm(dim=-1).mean().item()

            original_vocab_size = len(tokenizer)
            add_new_tokens(model, tokenizer, new_tokens=special_tokens)
            num_added = len(tokenizer) - original_vocab_size
            logger.info("Added %d special tokens (via Unsloth); vocab size now %d", num_added, len(tokenizer))
        else:
            tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
            model = AutoModelForCausalLM.from_pretrained(cfg.base_model, dtype=torch.bfloat16)

            # Captured before resizing -- the target norm new tokens get
            # rescaled to, so they start in the same magnitude range as the
            # rest of the vocabulary instead of the RQ-VAE codebook's native
            # (much smaller, near-zero) norm range.
            existing_norm_mean = model.get_input_embeddings().weight.norm(dim=-1).mean().item()

            original_vocab_size = len(tokenizer)
            num_added = tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
            model.resize_token_embeddings(len(tokenizer))
            logger.info("Added %d special tokens; vocab size now %d", num_added, len(tokenizer))

        # Explicit consistency check -- cheap insurance against exactly the
        # class of tied-embeddings/dimension-mismatch bug already hit once
        # in this project (see finetune_qwen.py's git history).
        vocab_size = len(tokenizer)
        input_size = model.get_input_embeddings().weight.shape[0]
        output_size = model.get_output_embeddings().weight.shape[0]
        if not (vocab_size == input_size == output_size):
            raise RuntimeError(
                f"Dimension mismatch after resize: tokenizer={vocab_size}, "
                f"input_embeddings={input_size}, output_embeddings={output_size}"
            )
        logger.info("Verified vocab_size == input_embeddings == output_embeddings == %d", vocab_size)

        # Overwrite HF's default mean/covariance random init for all 1026 new
        # sid tokens -- codebook-grounded where a real RQ-VAE vector exists
        # (768 L0-L2 tokens), distinct random vectors elsewhere (sid_start,
        # sid_end, L3) since the default was measured to collapse them into
        # a near-degenerate cluster. See _sid_token_init_vectors and the
        # module docstring.
        init_vectors = _sid_token_init_vectors(
            cfg.rqvae_checkpoint_path, model.config.hidden_size, existing_norm_mean, cfg.seed,
        )
        embedding_weight = model.get_input_embeddings().weight
        with torch.no_grad():
            for token, vector in init_vectors.items():
                token_id = tokenizer.convert_tokens_to_ids(token)
                embedding_weight[token_id] = vector.to(dtype=embedding_weight.dtype)
        logger.info("Applied grounded/distinct initialization to %d sid tokens", len(init_vectors))

        if cfg.load_in_4bit:
            # transformers' Trainer refuses to fine-tune a quantized model
            # unless it's wrapped in a PeftModel, even though the only
            # parameters we actually want trainable (embed_tokens/lm_head)
            # were never quantized in the first place (only Linear layers
            # get bitsandbytes-wrapped). modules_to_save does the real work
            # here, identically to the non-quantized path below; target_modules
            # is a single rank-1 adapter purely to satisfy peft's API (it
            # requires at least one LoRA target) -- negligible next to the
            # ~391M-parameter embedding matrix, so this stays effectively
            # embedding-only, matching this stage's actual intent.
            model = FastLanguageModel.get_peft_model(
                model,
                r=1, lora_alpha=1, target_modules=["q_proj"],
                modules_to_save=["embed_tokens", "lm_head"],
                ensure_weight_tying=True,
                use_gradient_checkpointing=False,
            )
        else:
            for param in model.parameters():
                param.requires_grad = False
            model.get_input_embeddings().weight.requires_grad = True
            model.get_output_embeddings().weight.requires_grad = True

        # Both branches above make the WHOLE embedding matrix trainable
        # (modules_to_save clones the entire module; requires_grad=True on
        # the whole weight does the same directly) -- not just the new sid
        # rows. Left alone, every ordinary vocabulary token that appears in
        # a training example's prompt (i.e. almost all of them, under
        # full-sequence loss) gets a real gradient update too, perturbing
        # the pretrained vocabulary's embeddings during this stage's
        # unusually high LR. STAR (arXiv, "Semantic-ID Token-Embedding
        # Alignment for Generative Recommenders") makes this exact point --
        # their alignment stage explicitly freezes the pretrained vocabulary
        # and updates only the new Semantic-ID rows. Registering a
        # gradient-masking hook here gets the same guarantee without giving
        # up the modules_to_save/full-matrix-requires_grad mechanics both
        # branches need for other reasons (PEFT's API, tied-embedding
        # bookkeeping).
        _freeze_pretrained_vocab_gradient(model, original_vocab_size)

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.info("Trainable parameters: %d / %d (%.4f%%)", trainable, total, 100 * trainable / total)

        model.config.use_cache = False  # required for training even without gradient checkpointing here

        self.model, self.tokenizer = model, tokenizer
        return model, tokenizer

    def _format_example(self, example: dict) -> dict:
        """Wrap a raw example into the Alpaca prompt/completion format used by SFTTrainer."""
        prompt = ALPACA_PROMPT.format(instruction=example["instruction"], input=example["input"])
        return {"prompt": prompt, "completion": example["output"]}

    def load_dataset(self):
        """Load the SFT jsonl, optionally filter to grounding-only, sample, and format."""
        cfg = self.config
        if not cfg.train_path.exists():
            raise FileNotFoundError(f"Train dataset not found at {cfg.train_path}. Run build_finetune_dataset.py first.")

        full_dataset = load_dataset("json", data_files={"train": cfg.train_path.as_posix()})["train"]

        # Captured from the FULL (unfiltered) dataset, even when
        # grounding_only restricts the actual training sample below --
        # sequential/asy examples won't exist in a grounding-only sample,
        # but the generation-check probe still wants one real example of
        # that shape to show whether the (untrained, in this stage) task
        # looks sane or not.
        self._sample_history = next(ex["input"] for ex in full_dataset if ex["task"] in ("sequential", "asy"))

        dataset = full_dataset
        if cfg.grounding_only:
            before = len(dataset)
            dataset = dataset.filter(lambda ex: ex["task"] in GROUNDING_TASKS)
            logger.info("grounding_only: restricted sample pool %d -> %d examples", before, len(dataset))

        dataset = dataset.shuffle(seed=cfg.seed).select(range(min(len(dataset), cfg.max_training_samples)))
        dataset = dataset.map(self._format_example)
        logger.info("Loaded %d sampled training examples (grounding_only=%s)", len(dataset), cfg.grounding_only)
        return dataset

    def build_trainer(self, dataset):
        """Configure SFTConfig + probes and return a ready-to-train SFTTrainer."""
        cfg = self.config
        args = SFTConfig(
            output_dir=cfg.output_dir.as_posix(),
            per_device_train_batch_size=cfg.micro_batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            max_steps=cfg.max_steps,
            learning_rate=cfg.learning_rate,
            lr_scheduler_type=cfg.lr_scheduler_type,
            warmup_steps=cfg.warmup_steps,
            optim=cfg.optimizer,
            weight_decay=cfg.weight_decay,
            max_length=cfg.max_seq_length,
            bf16=True,
            logging_steps=cfg.logging_steps,
            save_strategy="steps" if cfg.save_steps else "no",
            save_steps=cfg.save_steps or 500,  # ignored when save_strategy="no"
            save_total_limit=cfg.save_total_limit,
            report_to=[],
            seed=cfg.seed,
            # Loss over the full sequence (prompt + completion), not just the
            # completion -- matches eugeneyan's recipe. Sid tokens get
            # gradient signal every time they appear, including on the
            # prompt side, which matters most here since this stage's whole
            # job is maximizing exposure for the freshly-initialized tokens.
            completion_only_loss=False,
        )

        # Fill in a real semantic ID for the second probe (similar_item-
        # shaped) and a real multi-item history for the third (asy-shaped)
        # now that the dataset is loaded. sample_history comes from
        # load_dataset's unfiltered pass (see its docstring note) since
        # grounding_only means no sequential/asy examples survive in
        # `dataset` itself.
        sample_id = next(ex["output"] for ex in dataset if "<|sid_start|>" in ex["output"])
        probe_fallbacks = [sample_id, self._sample_history]
        probes = []
        fallback_idx = 0
        for instr, inp in GENERATION_PROBES:
            if inp is None:
                probes.append((instr, probe_fallbacks[fallback_idx]))
                fallback_idx += 1
            else:
                probes.append((instr, inp))

        return SFTTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            train_dataset=dataset,
            args=args,
            callbacks=[GenerationCheckCallback(self.tokenizer, probes, cfg.generation_check_steps)],
        )

    def train(self):
        """Load model, load dataset, build the trainer, and run training."""
        self.load_model()
        dataset = self.load_dataset()
        trainer = self.build_trainer(dataset)
        trainer.train()

        if self.config.load_in_4bit:
            # merge_and_unload() doesn't actually clear bitsandbytes'
            # quantization bookkeeping on the frozen backbone (only q_proj
            # had a LoRA delta to merge -- everything else stays a 4-bit
            # Params4bit tensor under the hood), so save_pretrained()
            # afterward hits a NotImplementedError in transformers'
            # revert_weight_conversion, deep in code neither this project
            # nor Unsloth controls. Don't attempt it: the trainer's own
            # periodic checkpointing already saves a complete, working
            # adapter checkpoint (proven -- that's what recovered a real
            # run after this exact save crashed at the very end). Stage 2
            # loads that adapter directly instead of expecting a merged,
            # standalone checkpoint.
            logger.info(
                "Quantized run: skipping merge+save (known-broken, see "
                "module comments). The trainer's last periodic checkpoint "
                "under %s/checkpoint-* is the complete, valid Stage 1 "
                "result -- point Stage 2 at that adapter checkpoint "
                "directly, not at %s itself.",
                self.config.output_dir, self.config.output_dir,
            )
            return

        self.model.save_pretrained(self.config.output_dir.as_posix())
        self.tokenizer.save_pretrained(self.config.output_dir.as_posix())
        logger.info("Saved warmed-up model + tokenizer to %s", self.config.output_dir)
        logger.info("Point Stage 2's FineTuneConfig.base_model at this path to continue from here.")


if __name__ == "__main__":
    EmbeddingWarmupTrainer(EmbeddingWarmupConfig()).train()