"""Stage 2 (QLoRA fine-tune) for the Qwen3-4B pivot.

Loads Stage 1's trained embed_tokens/lm_head weights directly onto a freshly
loaded 4-bit base model (not via PeftModel.from_pretrained + merge_and_unload,
which leaves stale quantization metadata that breaks Unsloth's fused LoRA
kernels and later save_pretrained() calls), then wraps with a new LoraConfig
for real task-specific adaptation. Saves as an adapter, never merged.
"""

from unsloth import FastLanguageModel  # isort: skip -- must import before transformers/trl/peft

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from datasets import load_dataset
from safetensors import safe_open
from transformers import AutoTokenizer, TrainerCallback
from trl import SFTConfig, SFTTrainer

from logger import Logger

logger = Logger.get_logger(__name__)

GENERATION_PROBES = [
    ("What is the semantic ID for this game?", "Half-Life 2"),
    ("A player enjoyed this game. Suggest another game they would likely also enjoy.", None),
    ("Given a user's game history, ordered from most to least played, predict the name of the next game they are likely to enjoy.", None),
]

LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


@dataclass
class QLoraFineTuneConfig:
    """Configuration for the Stage 2 QLoRA fine-tune on Qwen3-4B."""

    base_model_name: str = "Qwen/Qwen3-4B"
    stage1_adapter_path: Path = Path("outputs/qwen3-4b-embed-warmup/checkpoint-2250")
    data_dir: Path = Path("data")
    train_path: Optional[Path] = None
    val_path: Optional[Path] = None
    output_dir: Path = Path("outputs/qwen3-4b-qlora")
    max_seq_length: int = 192

    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0

    micro_batch_size: int = 1  # effective batch = micro_batch_size * gradient_accumulation_steps
    gradient_accumulation_steps: int = 128
    num_epochs: int = 3
    max_steps: Optional[int] = 1730  # capped to a ~24h training budget; None falls back to num_epochs
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    optimizer: str = "paged_adamw_8bit"
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    save_steps: int = 150
    save_total_limit: int = 5
    eval_steps: int = 500
    logging_steps: int = 10
    generation_check_steps: int = 300
    seed: int = 0
    resume_from_checkpoint: Optional[str] = None

    def __post_init__(self):
        """Fill in computed train/val paths and log a config summary."""
        if self.train_path is None:
            self.train_path = self.data_dir / "output" / "sft_train.jsonl"
        if self.val_path is None:
            self.val_path = self.data_dir / "output" / "sft_val.jsonl"

        logger.info(
            "QLoraFineTuneConfig: base_model=%s, stage1_adapter=%s, lora_r=%d, "
            "micro_batch_size=%d, grad_accum=%d, effective_batch=%d, epochs=%d, max_steps=%s, lr=%.2e",
            self.base_model_name, self.stage1_adapter_path, self.lora_r,
            self.micro_batch_size, self.gradient_accumulation_steps,
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


class QLoraFineTuneTrainer:
    """Stage 2 trainer: copy Stage 1's embeddings onto a fresh quantized base and train QLoRA on top."""

    def __init__(self, config: QLoraFineTuneConfig):
        """Store the config; load model + dataset on demand."""
        self.config = config
        self.model = None
        self.tokenizer = None

    def _load_stage1_embedding_weights(self, adapter_path: str):
        """Pull just the trained embed_tokens/lm_head tensors out of Stage 1's adapter checkpoint."""
        st_path = Path(adapter_path) / "adapter_model.safetensors"
        with safe_open(st_path.as_posix(), framework="pt") as f:
            embed = f.get_tensor("base_model.model.model.embed_tokens.modules_to_save.weight")
            lm_head = f.get_tensor("base_model.model.lm_head.modules_to_save.weight")
        return embed, lm_head

    def load_model(self):
        """Load fresh Qwen3-4B in 4-bit, copy Stage 1's embedding weights, and apply QLoRA. Returns (model, tokenizer)."""
        cfg = self.config
        adapter_path = cfg.stage1_adapter_path.resolve().as_posix()

        model, _ = FastLanguageModel.from_pretrained(
            model_name=cfg.base_model_name,
            max_seq_length=cfg.max_seq_length,
            dtype=torch.bfloat16,
            load_in_4bit=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        model.resize_token_embeddings(len(tokenizer))
        logger.info("Resized base model to Stage 1's vocab size: %d", len(tokenizer))

        embed_weight, lm_head_weight = self._load_stage1_embedding_weights(adapter_path)
        input_embeddings = model.get_input_embeddings()
        output_embeddings = model.get_output_embeddings()
        with torch.no_grad():
            input_embeddings.weight.copy_(embed_weight.to(input_embeddings.weight.dtype))
            output_embeddings.weight.copy_(lm_head_weight.to(output_embeddings.weight.dtype))
        logger.info("Copied Stage 1's trained embed_tokens/lm_head weights onto the fresh base model")

        model = FastLanguageModel.get_peft_model(
            model,
            r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
            target_modules=LORA_TARGET_MODULES,
            modules_to_save=["embed_tokens", "lm_head"],
            ensure_weight_tying=True,
            use_gradient_checkpointing="unsloth",
        )

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

    def train(self):
        """Load model, load dataset, build the trainer, run training, and save the adapter."""
        self.load_model()
        dataset = self.load_dataset()
        trainer = self.build_trainer(dataset)
        trainer.train(resume_from_checkpoint=self.config.resume_from_checkpoint)

        self.model.save_pretrained(self.config.output_dir.as_posix())
        self.tokenizer.save_pretrained(self.config.output_dir.as_posix())
        logger.info("Saved QLoRA adapter + tokenizer to %s", self.config.output_dir)


if __name__ == "__main__":
    QLoraFineTuneTrainer(QLoraFineTuneConfig()).train()