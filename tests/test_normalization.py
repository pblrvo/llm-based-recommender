"""Unit tests for l2norm / L2NormalizationLayer: output is unit-norm along
the given dimension, direction is preserved, and the eps guard keeps
zero-vector inputs finite instead of producing NaN/Inf."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from normalization import L2NormalizationLayer, l2norm


def test_l2norm_output_has_unit_norm():
    x = torch.randn(8, 16)
    out = l2norm(x)
    norms = out.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(8), atol=1e-5)


def test_l2norm_preserves_direction():
    x = torch.tensor([[3.0, 4.0]])  # norm = 5
    out = l2norm(x)
    assert torch.allclose(out, torch.tensor([[0.6, 0.8]]), atol=1e-6)


def test_l2norm_zero_vector_stays_finite_via_eps():
    x = torch.zeros(1, 4)
    out = l2norm(x, eps=1e-12)
    assert torch.isfinite(out).all()
    assert torch.equal(out, torch.zeros(1, 4))


def test_l2norm_respects_dim_argument():
    x = torch.randn(3, 5, 7)
    out = l2norm(x, dim=1)
    norms_along_dim1 = out.norm(dim=1)
    assert torch.allclose(norms_along_dim1, torch.ones(3, 7), atol=1e-5)


def test_l2normalization_layer_matches_functional_l2norm():
    x = torch.randn(4, 10)
    layer = L2NormalizationLayer(dim=-1, eps=1e-12)
    assert torch.allclose(layer(x), l2norm(x, dim=-1, eps=1e-12))


def test_l2normalization_layer_is_stateless_and_reusable():
    layer = L2NormalizationLayer()
    a = layer(torch.randn(2, 6))
    b = layer(torch.randn(2, 6))
    assert torch.allclose(a.norm(dim=-1), torch.ones(2), atol=1e-5)
    assert torch.allclose(b.norm(dim=-1), torch.ones(2), atol=1e-5)


@pytest.mark.parametrize("scale", [0.001, 1.0, 1000.0])
def test_l2norm_is_scale_invariant_in_direction(scale):
    x = torch.tensor([[1.0, 2.0, 2.0]])  # norm = 3
    out = l2norm(x * scale)
    expected = torch.tensor([[1 / 3, 2 / 3, 2 / 3]])
    assert torch.allclose(out, expected, atol=1e-4)
