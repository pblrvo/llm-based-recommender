"""Unit tests for VectorQuantizer: nearest-code lookup, loss computation,
usage tracking, and dead/dominant-code resets. Pure CPU torch with a tiny
codebook -- no GPU or real embeddings needed."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import RQVAEConfig
from vector_quantizer import VectorQuantizer


def make_vq(codebook_size=4, codebook_embedding_dim=3, commitment_weight=0.25) -> VectorQuantizer:
    config = RQVAEConfig(
        codebook_size=codebook_size,
        codebook_embedding_dim=codebook_embedding_dim,
        commitment_weight=commitment_weight,
    )
    return VectorQuantizer(config)


def test_find_nearest_codes_picks_the_closest_codebook_vector():
    vq = make_vq(codebook_size=3, codebook_embedding_dim=2)
    with torch.no_grad():
        vq.embedding.weight.copy_(torch.tensor([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]]))

    x = torch.tensor([[9.0, 1.0], [0.5, -0.5], [1.0, 9.0]])
    indices, quantized = vq.find_nearest_codes(x)

    assert indices.tolist() == [1, 0, 2]
    assert torch.allclose(quantized, vq.embedding.weight[indices])


def test_quantize_matches_find_nearest_codes():
    vq = make_vq()
    x = torch.randn(5, vq.codebook_embedding_dim)
    idx_a, q_a = vq.find_nearest_codes(x)
    idx_b, q_b = vq.quantize(x)
    assert torch.equal(idx_a, idx_b)
    assert torch.equal(q_a, q_b)


def test_forward_output_shapes():
    vq = make_vq(codebook_size=8, codebook_embedding_dim=4)
    x = torch.randn(6, 4)
    out = vq(x)
    assert out.quantized_st.shape == x.shape
    assert out.quantized.shape == x.shape
    assert out.indices.shape == (6,)
    assert out.loss.ndim == 0
    assert out.codebook_loss.ndim == 0
    assert out.commitment_loss.ndim == 0


def test_forward_loss_is_codebook_plus_weighted_commitment():
    vq = make_vq(commitment_weight=0.37)
    x = torch.randn(10, vq.codebook_embedding_dim)
    out = vq(x)
    expected_loss = out.codebook_loss + 0.37 * out.commitment_loss
    assert out.loss.item() == pytest.approx(expected_loss.item(), rel=1e-5)


def test_forward_eval_mode_quantized_st_equals_quantized_value():
    """In eval mode quantized_st is a straight-through passthrough (x + (q -
    x).detach()), which must equal quantized numerically even though the
    computation graph differs."""
    vq = make_vq()
    vq.eval()
    x = torch.randn(4, vq.codebook_embedding_dim)
    out = vq(x)
    assert torch.allclose(out.quantized_st, out.quantized, atol=1e-6)


def test_forward_training_mode_updates_usage_eval_mode_does_not():
    vq = make_vq(codebook_size=4)
    x = torch.randn(20, vq.codebook_embedding_dim)

    vq.eval()
    vq(x)
    assert vq.update_count.item() == 0
    assert vq.usage_count.sum().item() == 0

    vq.train()
    vq(x)
    assert vq.update_count.item() == 1
    assert vq.usage_count.sum().item() == 20


def test_get_usage_rate_before_any_update_is_zero():
    vq = make_vq()
    assert vq.get_usage_rate() == 0.0


def test_get_usage_rate_reflects_fraction_of_codes_touched():
    vq = make_vq(codebook_size=4, codebook_embedding_dim=2)
    with torch.no_grad():
        vq.embedding.weight.copy_(torch.tensor([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [30.0, 0.0]]))
    # All inputs land nearest code 0 or 1 only -- codes 2, 3 stay unused.
    x = torch.tensor([[0.1, 0.0], [9.9, 0.0], [0.2, 0.0], [10.1, 0.0]])
    vq.train()
    vq(x)
    assert vq.get_usage_rate() == pytest.approx(0.5)


def test_get_max_usage_share_before_any_update_is_zero():
    vq = make_vq()
    assert vq.get_max_usage_share() == 0.0


def test_get_max_usage_share_detects_dominant_code():
    vq = make_vq(codebook_size=2, codebook_embedding_dim=2)
    with torch.no_grad():
        vq.embedding.weight.copy_(torch.tensor([[0.0, 0.0], [100.0, 0.0]]))
    # 9 of 10 points land on code 0, 1 on code 1.
    x = torch.cat([torch.zeros(9, 2) + 0.01, torch.tensor([[100.0, 0.0]])])
    vq.train()
    vq(x)
    assert vq.get_max_usage_share() == pytest.approx(0.9)


def test_reset_usage_count_zeroes_counts():
    vq = make_vq()
    vq.train()
    vq(torch.randn(5, vq.codebook_embedding_dim))
    assert vq.usage_count.sum().item() > 0
    vq.reset_usage_count()
    assert vq.usage_count.sum().item() == 0


def test_reset_unused_codebook_vectors_replaces_dead_codes():
    vq = make_vq(codebook_size=4, codebook_embedding_dim=2)
    with torch.no_grad():
        vq.embedding.weight.copy_(torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]))
    # Only code 0 ever gets used.
    x = torch.zeros(10, 2) + 0.01
    vq.train()
    vq(x)
    assert vq.get_usage_rate() == pytest.approx(0.25)  # only 1/4 codes touched

    original_code_0 = vq.embedding.weight[0].clone()
    batch_data = torch.full((10, 2), 99.0)  # distinctive replacement source
    vq.reset_unused_codebook_vectors(batch_data)

    # The used code (0) must be untouched; the 3 dead codes must be replaced
    # with vectors drawn from batch_data.
    assert torch.equal(vq.embedding.weight[0], original_code_0)
    for i in (1, 2, 3):
        assert torch.allclose(vq.embedding.weight[i], torch.full((2,), 99.0))
    # Usage count is cleared after a reset pass.
    assert vq.usage_count.sum().item() == 0


def test_reset_unused_codebook_vectors_noop_before_any_update():
    vq = make_vq()
    original_weights = vq.embedding.weight.clone()
    vq.reset_unused_codebook_vectors(torch.randn(10, vq.codebook_embedding_dim))
    assert torch.equal(vq.embedding.weight, original_weights)


def test_reset_unused_codebook_vectors_resets_dominant_code_when_thresholded():
    vq = make_vq(codebook_size=2, codebook_embedding_dim=2)
    with torch.no_grad():
        vq.embedding.weight.copy_(torch.tensor([[0.0, 0.0], [100.0, 0.0]]))
    # 9 of 10 -> code 0 (dominant), 1 of 10 -> code 1 (still "used" so not dead).
    x = torch.cat([torch.zeros(9, 2) + 0.01, torch.tensor([[100.0, 0.0]])])
    vq.train()
    vq(x)
    assert vq.get_usage_rate() == pytest.approx(1.0)  # both codes technically used
    assert vq.get_max_usage_share() == pytest.approx(0.9)  # but code 0 dominates

    batch_data = torch.full((10, 2), -7.0)
    vq.reset_unused_codebook_vectors(batch_data, dominance_threshold=0.8)

    # Code 0 (90% share, over the 0.8 threshold) should be reset even though
    # it was never "dead" -- get_usage_rate() alone would have missed this.
    assert torch.allclose(vq.embedding.weight[0], torch.full((2,), -7.0))


def test_reset_unused_codebook_vectors_skips_when_batch_too_small():
    vq = make_vq(codebook_size=4, codebook_embedding_dim=2)
    x = torch.zeros(10, 2)
    with torch.no_grad():
        vq.embedding.weight.copy_(torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]))
    vq.train()
    vq(x)  # only code 0 used -> 3 dead codes need resetting

    original_weights = vq.embedding.weight.clone()
    too_small_batch = torch.randn(2, 2)  # fewer samples than the 3 codes needing reset
    vq.reset_unused_codebook_vectors(too_small_batch)

    # Should skip the reset entirely rather than partially resetting.
    assert torch.equal(vq.embedding.weight, original_weights)
