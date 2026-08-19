"""Stage 1 of a two-stage fine-tuning strategy: warm up the new semantic-ID
token embeddings before any task-specific (LoRA) training happens.

Freezes every parameter except embed_tokens/lm_head and trains only those, at
a high learning rate, on a data sample -- giving the new tokens a
differentiated starting point before Stage 2's real task-specific training.
Uses codebook-grounded initialization (`_codebook_grounded_vectors`) instead
of HF's default random init, gradient-masks the pretrained vocabulary
(`_freeze_pretrained_vocab_gradient`) so only the new sid tokens update, and
by default restricts the training sample to grounding tasks
(`EmbeddingWarmupConfig.grounding_only`).

Plain transformers/TRL by default (`load_in_4bit=False`). For models too
large to hold frozen in bf16 alongside trainable embeddings,
`load_in_4bit=True` switches to a quantized backbone via Unsloth's
FastLanguageModel.
"""

from unsloth import FastLanguageModel, add_new_tokens  # isort: skip -- must import before transformers/trl/peft

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

# The 4th sid level (<|sid_L3_*|>) is a collision-disambiguation digit, not a learned RQ-VAE level.
CODEBOOK_LEVELS = 3

GROUNDING_TASKS = {"grounding_id2name", "grounding_name2id"}


def _codebook_grounded_vectors(
    rqvae_checkpoint_path: Path, hidden_size: int, target_norm: float, generator: torch.Generator,
) -> Dict[str, torch.Tensor]:
    """Build init vectors for the L0-L2 sid tokens from the trained RQ-VAE codebooks.

    Projects each level's codebook into the LLM embedding space with a fixed
    isometric projection (orthonormal columns from QR-decomposing a seeded
    random Gaussian), preserving the codebook's relative geometry exactly,
    then rescales per level so norms land where the rest of the vocabulary lives.
    """
    state_dict = torch.load(rqvae_checkpoint_path, map_location="cpu", weights_only=False)["model_state_dict"]

    vectors: Dict[str, torch.Tensor] = {}
    for level in range(CODEBOOK_LEVELS):
        codebook = state_dict[f"vq_layers.{level}.embedding.weight"].float()
        num_codes, code_dim = codebook.shape

        projection, _ = torch.linalg.qr(torch.randn(hidden_size, code_dim, generator=generator))
        projected = codebook @ projection.T

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
    """Build init vectors for sid tokens with no real learned codebook to ground them in (sid_start/sid_end/L3).

    Independent random unit vectors, scaled to target_norm -- nearly orthogonal
    to each other with high probability in a high-dim space.
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
    """Register a backward hook that zeros gradients on pretrained vocab rows (< original_vocab_size).

    Zeroing the gradient rather than splitting the parameter means AdamW's
    decoupled weight decay would still silently shrink the "frozen" rows --
    that's why EmbeddingWarmupConfig.weight_decay defaults to 0.0.
    """
    seen = set()
    for get_embeddings in (model.get_input_embeddings, model.get_output_embeddings):
        weight = get_embeddings().weight
        if id(weight) in seen:
            continue
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

# Fixed probes for the live generation check. None inputs are filled in at runtime from the dataset.
GENERATION_PROBES = [
    ("What is the semantic ID for this game?", "Half-Life 2"),
    ("A player enjoyed this game. Suggest another game they would likely also enjoy.", None),
    (
        "Given a user's game history, ordered from most to least played, predict the name of the next game they are likely to enjoy.",
        None,
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
    load_in_4bit: bool = False  # for models too large to hold frozen in bf16 alongside trainable embeddings
    max_seq_length: int = 192

    max_training_samples: int = 20000  # a sample, not full convergence -- Stage 2 does that
    grounding_only: bool = True  # restrict to grounding_id2name/grounding_name2id (title<->sid pairs)

    learning_rate: float = 1e-3  # high relative to normal fine-tuning: only 2 freshly-initialized matrices train
    micro_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_steps: int = 750
    warmup_steps: int = 50
    weight_decay: float = 0.0  # nonzero decay would still shrink the gradient-masked "frozen" rows
    lr_scheduler_type: str = "cosine"
    optimizer: str = "adamw_torch"
    logging_steps: int = 10
    generation_check_steps: int = 150
    seed: int = 0

    save_steps: Optional[int] = None  # None = no intermediate checkpoints, only the final save_pretrained()
    save_total_limit: int = 3

    def __post_init__(self):
        """Fill in computed defaults and validate the checkpoint/quantization combination."""
        if self.train_path is None:
            self.train_path = self.data_dir / "output" / "sft_train.jsonl"
        if self.special_tokens_path is None:
            self.special_tokens_path = self.data_dir / "output" / "sft_special_tokens.json"

        if self.load_in_4bit and self.save_steps is None:
            raise ValueError(
                "load_in_4bit=True requires save_steps to be set -- the "
                "final save is skipped for quantized runs (see train()), "
                "so periodic checkpoints are the only output."
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
        """Load base model + tokenizer, add sid tokens, init their weights, and make them trainable. Returns (model, tokenizer)."""
        cfg = self.config
        special_tokens = self._load_special_tokens()

        if cfg.load_in_4bit:
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

            existing_norm_mean = model.get_input_embeddings().weight.norm(dim=-1).mean().item()

            original_vocab_size = len(tokenizer)
            num_added = tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
            model.resize_token_embeddings(len(tokenizer))
            logger.info("Added %d special tokens; vocab size now %d", num_added, len(tokenizer))

        vocab_size = len(tokenizer)
        input_size = model.get_input_embeddings().weight.shape[0]
        output_size = model.get_output_embeddings().weight.shape[0]
        if not (vocab_size == input_size == output_size):
            raise RuntimeError(
                f"Dimension mismatch after resize: tokenizer={vocab_size}, "
                f"input_embeddings={input_size}, output_embeddings={output_size}"
            )
        logger.info("Verified vocab_size == input_embeddings == output_embeddings == %d", vocab_size)

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
            # transformers' Trainer requires a PeftModel to fine-tune a quantized model, even
            # though embed_tokens/lm_head were never quantized; target_modules is a negligible
            # rank-1 placeholder purely to satisfy peft's API -- modules_to_save does the real work.
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

        # Both branches above make the WHOLE embedding matrix trainable, not just the new sid
        # rows; this masks the gradient so only the new rows actually update.
        _freeze_pretrained_vocab_gradient(model, original_vocab_size)

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.info("Trainable parameters: %d / %d (%.4f%%)", trainable, total, 100 * trainable / total)

        model.config.use_cache = False

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

        # Captured from the full (unfiltered) dataset since grounding_only would otherwise
        # exclude any sequential/asy example the generation-check probe wants to show.
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
            completion_only_loss=False,  # sid tokens get gradient signal on the prompt side too
        )

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
            # merge_and_unload() leaves stale bitsandbytes quantization metadata that breaks
            # save_pretrained(); the trainer's own periodic checkpoints are the valid output instead.
            logger.info(
                "Quantized run: skipping merge+save (known-broken). The trainer's last periodic "
                "checkpoint under %s/checkpoint-* is the complete, valid Stage 1 result -- point "
                "Stage 2 at that adapter checkpoint directly, not at %s itself.",
                self.config.output_dir, self.config.output_dir,
            )
            return

        self.model.save_pretrained(self.config.output_dir.as_posix())
        self.tokenizer.save_pretrained(self.config.output_dir.as_posix())
        logger.info("Saved warmed-up model + tokenizer to %s", self.config.output_dir)
        logger.info("Point Stage 2's FineTuneConfig.base_model at this path to continue from here.")


if __name__ == "__main__":
    EmbeddingWarmupTrainer(EmbeddingWarmupConfig()).train()