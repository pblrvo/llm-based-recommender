"""Unit/integration tests for RQVAE: forward-pass shapes, the
encode<->decode semantic-ID round trip, and the small pure-math metric
helpers (unique-ID proportion, codebook usage/max-share passthrough).
Tiny CPU-only config -- no GPU, no real item embeddings."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import RQVAEConfig
from encoder import MLP
from normalization import l2norm
from rqvae import RQVAE


def make_model(item_embedding_dim=16, codebook_embedding_dim=4, levels=3, codebook_size=8) -> RQVAE:
    config = RQVAEConfig(
        item_embedding_dim=item_embedding_dim,
        encoder_hidden_dims=[12, 8],
        codebook_embedding_dim=codebook_embedding_dim,
        codebook_quantization_levels=levels,
        codebook_size=codebook_size,
    )
    return RQVAE(config)


# ---------------------------------------------------------------------
# MLP (encoder.py) -- constructor/shape validation used by RQVAE's encoder/decoder
# ---------------------------------------------------------------------


def test_mlp_forward_output_shape():
    mlp = MLP(input_dim=10, hidden_dim=[6, 4], output_dim=3)
    out = mlp(torch.randn(5, 10))
    assert out.shape == (5, 3)


def test_mlp_rejects_wrong_input_dim():
    mlp = MLP(input_dim=10, hidden_dim=[6], output_dim=3)
    with pytest.raises(ValueError):
        mlp(torch.randn(5, 7))


def test_mlp_normalize_output_matches_manual_l2norm():
    """Checks normalize=True wires L2NormalizationLayer onto the MLP's raw
    output. Compares against manually re-normalizing the un-normalized MLP's
    output (same weights) rather than asserting unit-norm outright: ReLU can
    legitimately zero out an entire row by chance for some random input/
    weight combination, and a genuinely zero pre-norm vector correctly stays
    zero-norm post-normalization too (see test_normalization.py's own eps
    coverage) -- that's not a bug in the flag, just a real edge case."""
    torch.manual_seed(0)
    raw_mlp = MLP(input_dim=10, hidden_dim=[6], output_dim=4, normalize=False)
    normalized_mlp = MLP(input_dim=10, hidden_dim=[6], output_dim=4, normalize=True)
    normalized_mlp.load_state_dict(raw_mlp.state_dict(), strict=False)  # same weights, only the trailing layer differs

    x = torch.randn(5, 10)
    raw_out = raw_mlp(x)
    normalized_out = normalized_mlp(x)

    assert torch.allclose(normalized_out, l2norm(raw_out), atol=1e-6)


# ---------------------------------------------------------------------
# RQVAE forward pass
# ---------------------------------------------------------------------


def test_forward_output_shapes():
    model = make_model(item_embedding_dim=16, codebook_embedding_dim=4, levels=3)
    x = torch.randn(5, 16)
    x_recon, all_indices, loss_dict = model(x)

    assert x_recon.shape == x.shape
    assert len(all_indices) == 3  # one per quantization level
    for indices in all_indices:
        assert indices.shape == (5,)
    assert loss_dict["loss"].ndim == 0
    assert torch.isfinite(loss_dict["loss"])
    assert len(loss_dict["codebook_losses"]) == 3
    assert len(loss_dict["commitment_losses"]) == 3
    assert len(loss_dict["level_residuals"]) == 3


def test_forward_loss_equals_recon_plus_vq_loss():
    model = make_model()
    x = torch.randn(4, model.item_embedding_dim)
    _, _, loss_dict = model(x)
    assert loss_dict["loss"].item() == pytest.approx(
        (loss_dict["recon_loss"] + loss_dict["vq_loss"]).item(), rel=1e-5
    )


def test_encode_decode_round_trip_matches_forward_reconstruction_in_eval_mode():
    """In eval mode, forward()'s quantized_st is value-identical to the
    quantized codes (straight-through passthrough), so decoding the
    residual-quantization indices independently via encode_to_semantic_ids +
    decode_from_semantic_ids should reproduce forward()'s reconstruction
    exactly, not just approximately."""
    model = make_model(item_embedding_dim=12, codebook_embedding_dim=4, levels=2, codebook_size=8)
    model.eval()
    x = torch.randn(6, 12)

    x_recon, _, _ = model(x)
    semantic_ids = model.encode_to_semantic_ids(x)
    decoded = model.decode_from_semantic_ids(semantic_ids)

    assert torch.allclose(x_recon, decoded, atol=1e-5)


def test_encode_to_semantic_ids_shape_and_range():
    model = make_model(levels=3, codebook_size=8)
    x = torch.randn(7, model.item_embedding_dim)
    semantic_ids = model.encode_to_semantic_ids(x)

    assert semantic_ids.shape == (7, 3)
    assert semantic_ids.min().item() >= 0
    assert semantic_ids.max().item() < 8


def test_encode_to_semantic_ids_matches_forward_indices_in_eval_mode():
    model = make_model(levels=3, codebook_size=8)
    model.eval()
    x = torch.randn(5, model.item_embedding_dim)

    _, forward_indices, _ = model(x)
    forward_stack = torch.stack(forward_indices, dim=-1)

    semantic_ids = model.encode_to_semantic_ids(x)
    assert torch.equal(forward_stack, semantic_ids)


# ---------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------


def test_calculate_unique_ids_proportion_all_unique():
    model = make_model()
    ids = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    assert model.calculate_unique_ids_proportion(ids) == pytest.approx(1.0)


def test_calculate_unique_ids_proportion_with_duplicates():
    model = make_model()
    # Rows 0 and 2 are identical duplicates; row 1 is unique.
    ids = torch.tensor([[1, 2, 3], [4, 5, 6], [1, 2, 3]])
    assert model.calculate_unique_ids_proportion(ids) == pytest.approx(1 / 3)


def test_calculate_unique_ids_proportion_single_item_batch():
    model = make_model()
    ids = torch.tensor([[1, 2, 3]])
    assert model.calculate_unique_ids_proportion(ids) == pytest.approx(1.0)


def test_calculate_unique_ids_proportion_all_duplicates():
    model = make_model()
    ids = torch.tensor([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    assert model.calculate_unique_ids_proportion(ids) == pytest.approx(0.0)


def test_calculate_codebook_usage_and_max_share_track_underlying_vq_layers():
    model = make_model(levels=2, codebook_size=4)
    model.train()
    x = torch.randn(20, model.item_embedding_dim)
    model(x)  # populates usage stats for both levels

    usage = model.calculate_codebook_usage()
    max_share = model.calculate_codebook_max_share()

    assert len(usage) == 2 and len(max_share) == 2
    assert usage == [vq.get_usage_rate() for vq in model.vq_layers]
    assert max_share == [vq.get_max_usage_share() for vq in model.vq_layers]
    assert all(0.0 <= u <= 1.0 for u in usage)
    assert all(0.0 <= s <= 1.0 for s in max_share)


def test_calculate_avg_residual_norm_matches_manual_computation():
    model = make_model()
    residual = torch.randn(8, model.codebook_embedding_dim)
    expected = residual.norm(dim=-1).mean().item()
    assert model.calculate_avg_residual_norm(residual) == pytest.approx(expected)


def test_calculate_avg_residual_norm_zero_for_zero_residual():
    model = make_model()
    residual = torch.zeros(5, model.codebook_embedding_dim)
    assert model.calculate_avg_residual_norm(residual) == pytest.approx(0.0)
