"""Learning-rate schedule with an optional linear warmup phase.

Used by train_rqvae.py; not a torch.optim.lr_scheduler subclass since it's
simpler to drive from a manual step loop and to log/checkpoint directly.
"""

import math

from logger import Logger

logger = Logger.get_logger(__name__)

VALID_SCHEDULER_TYPES = ("cosine", "cosine_with_warmup")


class WarmupCosineScheduler:
    """Cosine learning-rate decay, optionally preceded by a linear warmup.

    scheduler_type="cosine": plain cosine decay from max_lr to min_lr over total_steps.
    scheduler_type="cosine_with_warmup": linear warmup from warmup_start_lr to max_lr
    over warmup_steps, then cosine decay from max_lr to min_lr over the remaining steps.
    """

    def __init__(
        self,
        optimizer,
        total_steps: int,
        max_lr: float,
        min_lr: float = 0.0,
        warmup_steps: int = 0,
        warmup_start_lr: float = 0.0,
        scheduler_type: str = "cosine_with_warmup",
    ):
        if scheduler_type not in VALID_SCHEDULER_TYPES:
            raise ValueError(f"Unknown scheduler_type: {scheduler_type!r}, expected one of {VALID_SCHEDULER_TYPES}")

        self.optimizer = optimizer
        self.total_steps = total_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps if scheduler_type == "cosine_with_warmup" else 0
        self.warmup_start_lr = warmup_start_lr
        self.scheduler_type = scheduler_type
        self._step_count = 0
        self._last_lr = self.warmup_start_lr if self.warmup_steps > 0 else max_lr

        logger.info(
            "WarmupCosineScheduler: type=%s, total_steps=%d, warmup_steps=%d, "
            "warmup_start_lr=%.2e, max_lr=%.2e, min_lr=%.2e",
            scheduler_type, total_steps, self.warmup_steps, warmup_start_lr, max_lr, min_lr,
        )

        self._set_lr(self._last_lr)

    def _compute_lr(self, step: int) -> float:
        if self.warmup_steps > 0 and step < self.warmup_steps:
            # Linear warmup
            progress = step / self.warmup_steps
            return self.warmup_start_lr + progress * (self.max_lr - self.warmup_start_lr)

        # Cosine decay over the post-warmup steps
        decay_steps = max(1, self.total_steps - self.warmup_steps)
        progress = min(1.0, (step - self.warmup_steps) / decay_steps)
        cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
        return self.min_lr + cosine_factor * (self.max_lr - self.min_lr)

    def _set_lr(self, lr: float) -> None:
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        self._last_lr = lr

    def step(self) -> float:
        """Advance one step and update the optimizer's learning rate. Returns the new LR."""
        lr = self._compute_lr(self._step_count)
        self._set_lr(lr)
        self._step_count += 1
        if self._step_count == self.warmup_steps:
            logger.info("Warmup complete at step %d, lr=%.2e", self._step_count, lr)
        return lr

    def get_last_lr(self) -> float:
        return self._last_lr

    def state_dict(self) -> dict:
        return {"step_count": self._step_count}

    def load_state_dict(self, state: dict) -> None:
        self._step_count = state["step_count"]
        # Re-apply the LR for the restored step immediately, otherwise the
        # optimizer keeps whatever LR was set at construction time (e.g.
        # warmup_start_lr) until the next .step() call.
        self._set_lr(self._compute_lr(self._step_count))
        logger.info("Resumed LR scheduler at step %d, lr=%.2e", self._step_count, self._last_lr)
