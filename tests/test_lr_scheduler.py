"""Unit tests for WarmupCosineScheduler: linear warmup ramp, cosine decay
shape, boundary conditions, and state_dict round-tripping. Pure CPU/math,
driven with a tiny real torch optimizer (one parameter) rather than a full
model."""

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lr_scheduler import WarmupCosineScheduler


def make_optimizer(lr: float = 0.0):
    param = torch.nn.Parameter(torch.zeros(1))
    return torch.optim.SGD([param], lr=lr)


def test_rejects_unknown_scheduler_type():
    with pytest.raises(ValueError):
        WarmupCosineScheduler(make_optimizer(), total_steps=10, max_lr=1.0, scheduler_type="linear")


def test_warmup_starts_at_warmup_start_lr_and_sets_optimizer_immediately():
    opt = make_optimizer()
    WarmupCosineScheduler(
        opt, total_steps=100, max_lr=1.0, warmup_steps=10, warmup_start_lr=0.01,
        scheduler_type="cosine_with_warmup",
    )
    # Constructor applies the initial LR without requiring a .step() call.
    assert opt.param_groups[0]["lr"] == pytest.approx(0.01)


def test_warmup_ramps_linearly_to_max_lr():
    opt = make_optimizer()
    sched = WarmupCosineScheduler(
        opt, total_steps=100, max_lr=1.0, warmup_steps=10, warmup_start_lr=0.0,
        scheduler_type="cosine_with_warmup",
    )
    lrs = [sched.step() for _ in range(10)]
    # step() at step_count=i returns the LR computed for step i, then advances.
    expected = [i / 10 for i in range(10)]
    assert lrs == pytest.approx(expected)


def test_lr_equals_max_lr_exactly_at_end_of_warmup():
    opt = make_optimizer()
    sched = WarmupCosineScheduler(
        opt, total_steps=100, max_lr=2.0, warmup_steps=5, warmup_start_lr=0.0,
        scheduler_type="cosine_with_warmup",
    )
    for _ in range(5):
        sched.step()
    lr_at_warmup_end = sched.step()  # step_count was 5 == warmup_steps
    assert lr_at_warmup_end == pytest.approx(2.0)


def test_cosine_decay_reaches_min_lr_at_final_step():
    opt = make_optimizer()
    total_steps = 50
    sched = WarmupCosineScheduler(
        opt, total_steps=total_steps, max_lr=1.0, min_lr=0.1, warmup_steps=0,
        scheduler_type="cosine",
    )
    # step() computes _compute_lr(step_count) *then* increments, so progress
    # reaches exactly 1.0 (step_count == total_steps) on the (total_steps+1)-th call.
    lr = None
    for _ in range(total_steps + 1):
        lr = sched.step()
    assert lr == pytest.approx(0.1, abs=1e-6)


def test_cosine_decay_is_monotonically_non_increasing_after_warmup():
    opt = make_optimizer()
    sched = WarmupCosineScheduler(
        opt, total_steps=50, max_lr=1.0, min_lr=0.0, warmup_steps=5, warmup_start_lr=0.0,
        scheduler_type="cosine_with_warmup",
    )
    lrs = [sched.step() for _ in range(50)]
    post_warmup = lrs[5:]
    assert all(a >= b - 1e-9 for a, b in zip(post_warmup, post_warmup[1:]))


def test_plain_cosine_ignores_warmup_steps():
    """scheduler_type='cosine' should behave as if warmup_steps=0 even if a
    nonzero value is passed in (see __init__: warmup_steps is zeroed out
    unless scheduler_type == 'cosine_with_warmup')."""
    opt = make_optimizer()
    sched = WarmupCosineScheduler(
        opt, total_steps=10, max_lr=1.0, min_lr=0.0, warmup_steps=5,
        scheduler_type="cosine",
    )
    assert sched.warmup_steps == 0
    first_lr = sched.step()
    # No warmup: first step() call should already be on the cosine curve, not
    # a warmup ramp toward max_lr.
    expected_first = 0.5 * (1 + math.cos(math.pi * (0 / 10)))
    assert first_lr == pytest.approx(expected_first)


def test_state_dict_round_trip_continues_the_same_lr_sequence():
    """The property that actually matters for reproducible training: a run
    interrupted and resumed at step 37 produces the same future LR sequence
    as one that ran uninterrupted. (load_state_dict's immediate LR set
    matches _compute_lr(step_count) -- the *upcoming* step's value, per its
    own docstring -- not get_last_lr()'s last-*returned* value, which is one
    step behind; only the forward sequence is asserted here.)"""

    def build(opt):
        return WarmupCosineScheduler(
            opt, total_steps=100, max_lr=1.0, min_lr=0.0, warmup_steps=10, warmup_start_lr=0.0,
            scheduler_type="cosine_with_warmup",
        )

    baseline = build(make_optimizer())
    to_checkpoint = build(make_optimizer())
    for _ in range(37):
        baseline.step()
        to_checkpoint.step()
    state = to_checkpoint.state_dict()

    resumed = build(make_optimizer())
    resumed.load_state_dict(state)

    # Neither `baseline` nor `resumed` has taken a step past 37 yet here, so
    # their next N calls should agree exactly.
    for _ in range(10):
        assert resumed.step() == pytest.approx(baseline.step())


def test_get_last_lr_reflects_most_recent_step():
    opt = make_optimizer()
    sched = WarmupCosineScheduler(
        opt, total_steps=10, max_lr=1.0, min_lr=0.0, warmup_steps=0, scheduler_type="cosine",
    )
    lr = sched.step()
    assert sched.get_last_lr() == pytest.approx(lr)
