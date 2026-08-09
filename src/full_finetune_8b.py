"""Stage 2 (full-parameter fine-tune) for the Qwen3-8B RunPod run -- same
rebalanced-data recipe as qlora_finetune.py, but every parameter is
trainable instead of a rank-8 LoRA adapter over a 4-bit base.

Feasible here specifically because the token budget is small: ~396K
examples at a measured mean of 64 tokens/example (well under the 192-token
cap) is ~24M tokens/epoch, ~48M for 2 epochs -- full fine-tuning an 8B model
over that little data is affordable on a single 80GB GPU with an 8-bit
optimizer, unlike this project's original 12GB local GPU (which forced
QLoRA for even a 4B model). Full fine-tuning is preferred over QLoRA here,
not just tolerated: teaching ~1,000 new semantic-ID tokens real grounding
across every layer is exactly the kind of representation shift a low-rank
adapter is limited in reshaping, and full-parameter updates should converge
more completely.

Loads Stage 1's output directly (see warmup_embeddings.py's `load_in_4bit=
False` path -- the one this script's Stage 1 uses for the same reason: 8B
fits unquantized). Unlike qlora_finetune.py, Stage 1's checkpoint here is
already a complete, plain HF model (extended vocab, warmed-up embeddings) --
not a PEFT adapter -- so there's no embedding-tensor-copying step required,
just load it and unfreeze every parameter.

Learning rate (2e-5) matches this project's own prior full-parameter
precedent (see qlora_finetune.py's docstring: "was 2e-5 for full_finetune.
py's full-parameter updates", from an earlier Qwen3-0.6B full fine-tune).
Optimizer is 8-bit AdamW (bitsandbytes, already a project dependency) --
the lever that keeps 8B full fine-tuning inside a single 80GB GPU's memory
instead of needing multi-GPU FSDP.

Batch size, gradient checkpointing memory headroom, and realistic
steps/second have NOT been smoke-tested on real 8B-scale hardware (this
project's other configs each cite specific measured numbers from a smoke
test on the target GPU -- see e.g. qlora_finetune.py's lora_r comment).
Run a short smoke test (a few hundred steps) on the actual RunPod instance
before committing to a full run, and adjust micro_batch_size/
gradient_accumulation_steps if memory is tighter or looser than expected.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import SFTConfig, SFTTrainer

from logger import Logger

logger = Logger.get_logger(__name__)

GENERATION_PROBES = [
    ("What is the semantic ID for this game?", "Half-Life 2"),
    ("A player enjoyed this game. Suggest another game they would likely also enjoy.", None),
    ("Given a user's game history, ordered from most to least played, predict the name of the next game they are likely to enjoy.", None),
]


@dataclass
class FullFineTuneConfig:
    """Configuration for the Stage 2 full-parameter fine-tune on Qwen3-8B."""

    # Stage 1's output directory (see warmup_embeddings.py, load_in_4bit=
    # False path) -- a complete model, not an adapter, since the non-
    # quantized path never wraps with PEFT. Point this at Stage 1's actual
    # output dir, e.g. outputs/qwen3-8b-embed-warmup (its final
    # save_pretrained(), not a numbered checkpoint -- unlike the 4-bit path,
    # the plain path's final save is NOT a known-broken no-op).
    stage1_model_path: Path = Path("outputs/qwen3-8b-embed-warmup")
    data_dir: Path = Path("data")
    train_path: Optional[Path] = None
    val_path: Optional[Path] = None
    output_dir: Path = Path("outputs/qwen3-8b-full-finetune")
    max_seq_length: int = 192

    # Effective batch = micro_batch_size * gradient_accumulation_steps.
    # NOT smoke-tested (see module docstring) -- this project's other
    # per-stage defaults were each set from a measured OOM/near-OOM point
    # on the target GPU; this one is a reasoned starting guess for an 80GB
    # card with full-parameter gradients + activations at a short (192-
    # token) sequence length, not a measured one. Validate on RunPod first.
    micro_batch_size: int = 4
    gradient_accumulation_steps: int = 32  # effective batch 128, matching this project's existing convention
    num_epochs: int = 2
    max_steps: Optional[int] = None  # None -> num_epochs drives training length (budget allows full epochs now)
    learning_rate: float = 2e-5  # full-parameter LR, matches this project's own prior full_finetune.py precedent
    warmup_ratio: float = 0.03
    optimizer: str = "paged_adamw_8bit"  # bitsandbytes 8-bit Adam -- keeps this fitting a single 80GB GPU
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    gradient_checkpointing: bool = True  # trades compute for memory -- every layer is trainable here, unlike LoRA
    save_steps: int = 300
    save_total_limit: int = 5
    eval_steps: int = 300
    logging_steps: int = 10
    generation_check_steps: int = 300
    seed: int = 0
    resume_from_checkpoint: Optional[str] = None

    # None (default) skips the push entirely -- set to a real repo id (e.g.
    # "pblrvo/Qwen3-8B-Game-semantic-IDs-v3") to upload the final model
    # after training. Requires a Hugging Face token with write access to be
    # available (HF_TOKEN env var, or a prior `huggingface-cli login`) --
    # checked at the START of train(), before the multi-hour run, not after
    # it -- a bad/missing token should fail fast, not surface only once
    # there's a finished model with nowhere to put it.
    hf_repo_id: Optional[str] = None

    def __post_init__(self):
        """Fill in computed train/val paths and log a config summary."""
        if self.train_path is None:
            self.train_path = self.data_dir / "output" / "sft_train.jsonl"
        if self.val_path is None:
            self.val_path = self.data_dir / "output" / "sft_val.jsonl"

        logger.info(
            "FullFineTuneConfig: stage1_model=%s, micro_batch_size=%d, grad_accum=%d, "
            "effective_batch=%d, epochs=%d, max_steps=%s, lr=%.2e",
            self.stage1_model_path, self.micro_batch_size, self.gradient_accumulation_steps,
            self.micro_batch_size * self.gradient_accumulation_steps,
            self.num_epochs, self.max_steps, self.learning_rate,
        )


class GenerationCheckCallback(TrainerCallback):
    """Periodically run fixed probes through the model for live signal during long runs."""

    def __init__(self, tokenizer, probes, interval: int):
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
                messages = [{"role": "user", "content": f"{instruction}\n{user_input}"}]
                prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
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


class FullFineTuneTrainer:
    """Stage 2 trainer: load Stage 1's warmed-up model directly and train every parameter."""

    def __init__(self, config: FullFineTuneConfig):
        """Store the config; load model + dataset on demand."""
        self.config = config
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Load Stage 1's complete model/tokenizer and unfreeze every parameter.

        Returns:
            (model, tokenizer) pair ready for SFTTrainer.
        """
        cfg = self.config
        stage1_path = cfg.stage1_model_path.resolve().as_posix()

        tokenizer = AutoTokenizer.from_pretrained(stage1_path)
        model = AutoModelForCausalLM.from_pretrained(stage1_path, dtype=torch.bfloat16)
        logger.info("Loaded Stage 1 model from %s (vocab size %d)", stage1_path, len(tokenizer))

        # Stage 1 leaves everything but embed_tokens/lm_head frozen
        # (requires_grad=False) -- undo that here, this stage trains every
        # parameter.
        for param in model.parameters():
            param.requires_grad = True

        if cfg.gradient_checkpointing:
            model.gradient_checkpointing_enable()
        model.config.use_cache = False  # required for training, doubly so with gradient checkpointing

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.info("Trainable parameters: %d / %d (%.2f%%)", trainable, total, 100 * trainable / total)

        self.model, self.tokenizer = model, tokenizer
        return model, tokenizer

    def _to_prompt_completion(self, example: dict) -> dict:
        """Convert a raw example into ChatML-style prompt + completion for SFTTrainer."""
        messages = [{"role": "user", "content": f"{example['instruction']}\n{example['input']}"}]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        completion = example["output"] + self.tokenizer.eos_token
        return {"prompt": prompt, "completion": completion}

    def load_dataset(self):
        """Load the train+val SFT jsonl files, capture a probe-history example, and format for SFTTrainer."""
        cfg = self.config
        dataset = load_dataset("json", data_files={
            "train": cfg.train_path.as_posix(), "validation": cfg.val_path.as_posix(),
        })

        self._sample_history = next(
            ex["input"] for ex in dataset["train"] if ex["task"] in ("sequential", "asy")
        )

        dataset = dataset.map(self._to_prompt_completion, remove_columns=dataset["train"].column_names)
        logger.info(
            "Loaded local dataset from %s: %d train, %d val examples",
            cfg.train_path, len(dataset["train"]), len(dataset["validation"]),
        )
        return dataset

    def build_trainer(self, dataset):
        """Configure SFTConfig + probes and return a ready-to-train SFTTrainer."""
        cfg = self.config
        args = SFTConfig(
            output_dir=cfg.output_dir.as_posix(),
            per_device_train_batch_size=cfg.micro_batch_size,
            per_device_eval_batch_size=cfg.micro_batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            num_train_epochs=cfg.num_epochs,
            max_steps=cfg.max_steps if cfg.max_steps is not None else -1,
            learning_rate=cfg.learning_rate,
            lr_scheduler_type=cfg.lr_scheduler_type,
            warmup_ratio=cfg.warmup_ratio,
            optim=cfg.optimizer,
            weight_decay=cfg.weight_decay,
            gradient_checkpointing=cfg.gradient_checkpointing,
            max_length=cfg.max_seq_length,
            bf16=True,
            eval_strategy="steps",
            eval_steps=cfg.eval_steps,
            save_strategy="steps",
            save_steps=cfg.save_steps,
            save_total_limit=cfg.save_total_limit,
            logging_steps=cfg.logging_steps,
            report_to=["tensorboard"],
            seed=cfg.seed,
            completion_only_loss=False,  # matches Stage 1 and qlora_finetune.py's full-sequence loss
        )

        sample_id = next(
            ex["completion"] for ex in dataset["train"] if "<|sid_start|>" in ex["completion"]
        )
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
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            args=args,
            callbacks=[GenerationCheckCallback(self.tokenizer, probes, cfg.generation_check_steps)],
        )

    def _check_hf_auth(self):
        """Fail fast if hf_repo_id is set but no valid write-access token is available.

        Checked before training starts, not after -- a missing/bad token
        should be caught in seconds, not discovered only once there's a
        finished model with nowhere to push it.
        """
        from huggingface_hub import HfApi

        try:
            HfApi().whoami()
        except Exception as e:
            raise RuntimeError(
                f"hf_repo_id={self.config.hf_repo_id!r} is set but no valid Hugging Face token was found. "
                "Set HF_TOKEN or run `huggingface-cli login` before starting training."
            ) from e

    def train(self):
        """Load model, load dataset, build the trainer, run training, save, and optionally push to the Hub."""
        if self.config.hf_repo_id:
            self._check_hf_auth()

        self.load_model()
        dataset = self.load_dataset()
        trainer = self.build_trainer(dataset)
        trainer.train(resume_from_checkpoint=self.config.resume_from_checkpoint)

        self.model.save_pretrained(self.config.output_dir.as_posix())
        self.tokenizer.save_pretrained(self.config.output_dir.as_posix())
        logger.info("Saved fully fine-tuned model + tokenizer to %s", self.config.output_dir)

        if self.config.hf_repo_id:
            logger.info("Pushing model + tokenizer to Hugging Face Hub: %s", self.config.hf_repo_id)
            self.model.push_to_hub(self.config.hf_repo_id)
            self.tokenizer.push_to_hub(self.config.hf_repo_id)
            logger.info("Pushed to https://huggingface.co/%s", self.config.hf_repo_id)


if __name__ == "__main__":
    FullFineTuneTrainer(FullFineTuneConfig()).train()
