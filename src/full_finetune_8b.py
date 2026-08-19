"""Stage 2 (full-parameter fine-tune) for the Qwen3-8B RunPod run: loads
Stage 1's warmed-up model and trains every parameter (unlike qlora_finetune.py's
rank-8 LoRA adapter), using an 8-bit AdamW optimizer to fit a single 80GB GPU.
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

    stage1_model_path: Path = Path("outputs/qwen3-8b-embed-warmup")
    data_dir: Path = Path("data")
    train_path: Optional[Path] = None
    val_path: Optional[Path] = None
    output_dir: Path = Path("outputs/qwen3-8b-full-finetune")
    max_seq_length: int = 192

    micro_batch_size: int = 8  # effective batch = micro_batch_size * gradient_accumulation_steps
    gradient_accumulation_steps: int = 16
    num_epochs: int = 2
    max_steps: Optional[int] = None
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.03
    optimizer: str = "paged_adamw_8bit"
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    gradient_checkpointing: bool = True
    save_steps: int = 300
    save_total_limit: int = 5
    eval_steps: int = 300
    eval_sample_size: int = 500  # in-training eval subsample; run_eval() does the full post-training eval
    logging_steps: int = 10
    generation_check_steps: int = 300
    seed: int = 0
    resume_from_checkpoint: Optional[str] = None

    # None skips the Hugging Face push entirely; otherwise a repo id to upload the final model to.
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
        """Load Stage 1's complete model/tokenizer and unfreeze every parameter. Returns (model, tokenizer)."""
        cfg = self.config
        stage1_path = cfg.stage1_model_path.resolve().as_posix()

        tokenizer = AutoTokenizer.from_pretrained(stage1_path)
        model = AutoModelForCausalLM.from_pretrained(stage1_path, dtype=torch.bfloat16)
        logger.info("Loaded Stage 1 model from %s (vocab size %d)", stage1_path, len(tokenizer))

        for param in model.parameters():
            param.requires_grad = True

        if cfg.gradient_checkpointing:
            model.gradient_checkpointing_enable()
        model.config.use_cache = False

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

        full_val_size = len(dataset["validation"])
        if full_val_size > cfg.eval_sample_size:
            dataset["validation"] = dataset["validation"].shuffle(seed=cfg.seed).select(range(cfg.eval_sample_size))
            logger.info(
                "Subsampled in-training eval set: %d -> %d examples (full val set is still used by "
                "run_eval()'s post-training evaluation)", full_val_size, cfg.eval_sample_size,
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
            completion_only_loss=False,
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
        """Fail fast if hf_repo_id is set but no valid write-access token is available."""
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
