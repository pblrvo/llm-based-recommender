from unsloth import FastLanguageModel  # isort: skip -- must import before transformers/trl/peft, see warmup_embeddings.py

"""Stage 2 (QLoRA fine-tune) for the Qwen3-4B pivot -- same rebalanced-data
recipe as full_finetune.py (Qwen3-0.6B, full-parameter), adapted for a model
too large to fully fine-tune on this 12GB GPU (~4B raw params needs ~24GB+
just for weights+gradients+optimizer states in the full_finetune.py recipe).
See the "4B vs 8B" conversation this session for why 4B was chosen over 8B:
QLoRA is required either way at this GPU's size, and 4B's tied embeddings
(vs. 8B's untied) meaningfully lower the trainable-embedding memory cost.

Loads Stage 1's adapter checkpoint (see warmup_embeddings.py,
`load_in_4bit=True` path) by copying its trained embed_tokens/lm_head
weights directly onto a freshly-loaded quantized base model (NOT via
PeftModel.from_pretrained + merge_and_unload -- see below for why), then
wraps with a NEW LoraConfig for real task-specific adaptation across
attention + MLP layers. Stage 1's own LoRA config only had a rank-1
q_proj delta (a placeholder to satisfy peft's API, not meant for real
adaptation) plus modules_to_save for embed_tokens/lm_head; that rank-1
delta is discarded here since Stage 2 applies its own real-rank LoRA to
q_proj anyway, which dominates it completely.

Why not PeftModel.from_pretrained + merge_and_unload(), as originally
tried: merge_and_unload() dequantizes+merges the (trivial, rank-1)
adapted layers back into plain bf16 nn.Linear, but leaves the model's
top-level quantization metadata (is_quantized, quantization_config,
hf_quantizer) untouched -- the same stale-metadata issue that breaks
save_pretrained() after merging (see below). Wrapping THAT inconsistent
model with a fresh get_peft_model() call causes Unsloth's fused LoRA
kernels to pick the wrong forward/backward code path (they trust the
stale "quantized" metadata), producing a dtype mismatch
(`bfloat16 != float`) inside unsloth/kernels/fast_lora.py's backward
pass. Directly copying just the trained embedding tensors onto a
cleanly-loaded quantized model sidesteps this entirely -- no merge, no
metadata inconsistency, and every step here (fresh 4-bit load, manual
embedding-weight overwrite) is independently already proven to work
elsewhere in this project (see warmup_embeddings.py's own codebook-
grounded init, which does the same kind of manual weight write).

Saves as an adapter (`model.save_pretrained()` on the PeftModel directly),
never attempting a merge-then-save -- merge_and_unload() doesn't actually
clear bitsandbytes' quantization bookkeeping on the frozen backbone (only
the LoRA-adapted layers had anything to merge), so a subsequent
save_pretrained() hits a NotImplementedError in transformers' internal
weight-conversion-reversal code. That's how Stage 1's very first run
crashed at its final step -- training itself completed fine, only the
save did not. Loading this script's own output later means base model +
PeftModel.from_pretrained(adapter_path), the same pattern used here to
load Stage 1's output.

Otherwise identical to full_finetune.py: local rebalanced dataset
(data/output/sft_train.jsonl / sft_val.jsonl), full-sequence loss
(completion_only_loss=False), ChatML formatting via the base model's own
chat template.
"""

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

# Standard Llama-style projection names Qwen3 uses -- the same 7 targets a
# typical QLoRA setup covers (attention: q/k/v/o, MLP: gate/up/down).
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


@dataclass
class QLoraFineTuneConfig:
    base_model_name: str = "Qwen/Qwen3-4B"
    # Stage 1's adapter checkpoint (see warmup_embeddings.py) -- the trainer's
    # own periodic checkpoint, not a merged standalone model (Stage 1's final
    # merge+save is a known no-op for quantized runs, see module docstring).
    stage1_adapter_path: Path = Path("outputs/qwen3-4b-embed-warmup/checkpoint-2250")
    data_dir: Path = Path("data")
    train_path: Optional[Path] = None
    val_path: Optional[Path] = None
    output_dir: Path = Path("outputs/qwen3-4b-qlora")
    max_seq_length: int = 192

    # Smoke-tested at r=16 first: reserved memory hit 12.14GB on a 12.288GB
    # card (nvidia-smi confirmed 12055/12288MiB used, 45% util, and the run
    # stalled for minutes on step 3) -- the same GPU shared-memory fallback
    # slowdown this project has hit before. Dropped to r=8 (still a
    # reasonable QLoRA rank) and switched to paged_adamw_8bit below for
    # real headroom instead of sitting at the edge.
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0

    # Effective batch = micro_batch_size * gradient_accumulation_steps.
    # Start as conservative as Stage 1 needed to be (micro_batch=1) given
    # Stage 2 trains far more parameters (real LoRA across 7 projections x
    # all layers, not just embed_tokens/lm_head) -- validate via smoke test
    # before a real run, same as every other run this project.
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 128  # effective batch 128, matching full_finetune.py
    num_epochs: int = 3
    # Smoke test measured 501.2s/10 steps = ~0.39s/micro-batch -- at 3 full
    # epochs over the 299,491-example dataset (effective batch 128), that's
    # ~7020 steps, ~97 hours. Capped to a ~24h budget instead (~1730 steps,
    # ~220K examples of exposure, ~0.74 epochs), matching Stage 1's own
    # approach of bounding by max_steps rather than running full epochs.
    # None means fall back to num_epochs (HF's max_steps=-1 default).
    max_steps: Optional[int] = 1730
    learning_rate: float = 2e-4  # LoRA-appropriate LR (was 2e-5 for full_finetune.py's full-parameter updates)
    warmup_ratio: float = 0.03
    optimizer: str = "paged_adamw_8bit"  # pages optimizer state to CPU under memory pressure instead of hard-failing/stalling
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
    """Same purpose as full_finetune.py's version: live signal every N
    steps instead of waiting for a multi-hour run to finish."""

    def __init__(self, tokenizer, probes, interval: int):
        self.tokenizer = tokenizer
        self.probes = probes
        self.interval = interval

    def _run(self, model, step: int):
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
        self._run(model, 0)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step > 0 and state.global_step % self.interval == 0:
            self._run(model, state.global_step)


class QLoraFineTuneTrainer:
    def __init__(self, config: QLoraFineTuneConfig):
        self.config = config
        self.model = None
        self.tokenizer = None

    def _load_stage1_embedding_weights(self, adapter_path: str):
        """Pull just the trained embed_tokens/lm_head tensors out of Stage
        1's adapter checkpoint (peft's modules_to_save clones), without
        going through PeftModel/merge_and_unload -- see module docstring."""
        st_path = Path(adapter_path) / "adapter_model.safetensors"
        with safe_open(st_path.as_posix(), framework="pt") as f:
            embed = f.get_tensor("base_model.model.model.embed_tokens.modules_to_save.weight")
            lm_head = f.get_tensor("base_model.model.lm_head.modules_to_save.weight")
        return embed, lm_head

    def load_model(self):
        cfg = self.config
        adapter_path = cfg.stage1_adapter_path.resolve().as_posix()

        model, _ = FastLanguageModel.from_pretrained(
            model_name=cfg.base_model_name,
            max_seq_length=cfg.max_seq_length,
            dtype=torch.bfloat16,
            load_in_4bit=True,
        )

        # Stage 1's adapter was trained against its own (vocab-extended)
        # tokenizer -- load it and resize the fresh base model to match
        # before writing in Stage 1's trained embedding weights.
        tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        model.resize_token_embeddings(len(tokenizer))
        logger.info("Resized base model to Stage 1's vocab size: %d", len(tokenizer))

        embed_weight, lm_head_weight = self._load_stage1_embedding_weights(adapter_path)
        input_embeddings = model.get_input_embeddings()
        output_embeddings = model.get_output_embeddings()
        with torch.no_grad():
            input_embeddings.weight.copy_(embed_weight.to(input_embeddings.weight.dtype))
            output_embeddings.weight.copy_(lm_head_weight.to(output_embeddings.weight.dtype))
        logger.info(
            "Copied Stage 1's trained embed_tokens/lm_head weights onto the fresh base model "
            "(discarded Stage 1's placeholder rank-1 q_proj delta -- negligible, see module docstring)"
        )

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
        messages = [{"role": "user", "content": f"{example['instruction']}\n{example['input']}"}]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        completion = example["output"] + self.tokenizer.eos_token
        return {"prompt": prompt, "completion": completion}

    def load_dataset(self):
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
        self.load_model()
        dataset = self.load_dataset()
        trainer = self.build_trainer(dataset)
        trainer.train(resume_from_checkpoint=self.config.resume_from_checkpoint)

        # Save as an adapter, not a merged model -- see module docstring for
        # why merge-then-save is broken for quantized models here. The
        # trainer's own periodic checkpoints (save_steps) are the same safe
        # format, so this is consistent with what already survives a crash.
        self.model.save_pretrained(self.config.output_dir.as_posix())
        self.tokenizer.save_pretrained(self.config.output_dir.as_posix())
        logger.info("Saved QLoRA adapter + tokenizer to %s", self.config.output_dir)


if __name__ == "__main__":
    QLoraFineTuneTrainer(QLoraFineTuneConfig()).train()
