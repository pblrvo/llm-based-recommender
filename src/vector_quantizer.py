"""Vector quantization layer with rotation-trick straight-through gradient.

Used as a building block by `rqvae.py` to produce hierarchical discrete
codes (the semantic IDs) from continuous embeddings.
"""

from normalization import l2norm
from typing import NamedTuple, Optional, Tuple
from torch import nn
from torch.nn import functional as F
from torch import Tensor
import torch

from config import RQVAEConfig
from logger import Logger

logger = Logger.get_logger(__name__)


class QuantizationOutput(NamedTuple):
    """Bundle of tensors returned by `VectorQuantizer.forward`: STE output, chosen codes, indices, and losses."""

    quantized_st: Tensor
    quantized: Tensor
    indices: Tensor
    loss: Tensor
    codebook_loss: Optional[Tensor]
    commitment_loss: Tensor


class VectorQuantizer(nn.Module):
    """Single-level vector quantizer with codebook usage tracking and dead-code reset."""

    def __init__(self, config: RQVAEConfig):
        """Set up the codebook and usage-tracking buffers."""
        super().__init__()
        self.codebook_embedding_dim = config.codebook_embedding_dim
        self.codebook_size = config.codebook_size
        self.commitment_weight = config.commitment_weight

        self.embedding = nn.Embedding(self.codebook_size, self.codebook_embedding_dim)
        self.embedding.weight.data.uniform_(-1 / self.codebook_size, 1 / self.codebook_size)

        self.register_buffer("usage_count", torch.zeros(self.codebook_size))
        self.register_buffer("update_count", torch.tensor(0))

        logger.info(
            "VectorQuantizer initialized: codebook_size=%d, codebook_dim=%d, commitment_weight=%.3f",
            self.codebook_size, self.codebook_embedding_dim, self.commitment_weight,
        )

    @staticmethod
    def safe_div(num: Tensor, den: Tensor, eps: float = 1e-6) -> Tensor:
        """Divide `num` by `den`, clamping the denominator to at least `eps`."""
        return num / den.clamp(min=eps)

    @staticmethod
    def rotation_trick(u: Tensor, q: Tensor, e: Tensor) -> Tensor:
        """Apply the rotation-trick straight-through estimator from arXiv:2410.06424.

        Returns a rotated version of encoder output `e` that equals `q` in the
        forward pass but carries gradients with respect to `e`.
        """
        w = l2norm(u + q, dim=-1, eps=1e-6).detach()

        w_col = w.unsqueeze(-1)
        w_row = w.unsqueeze(-2)
        u_col = u.unsqueeze(-1).detach()
        q_row = q.unsqueeze(-2).detach()

        if e.ndim == 2:
            e_expanded = e.unsqueeze(1)
            result = e_expanded - 2 * (e_expanded @ w_col @ w_row) + 2 * (e_expanded @ u_col @ q_row)
            return result.squeeze(1)
        else:
            return e - 2 * (e @ w_col @ w_row).squeeze(-1) + 2 * (e @ u_col @ q_row).squeeze(-1)

    @staticmethod
    def rotate_to(src: Tensor, tgt: Tensor) -> Tensor:
        """Apply the rotation-trick STE so the model can learn through the VQ layer.

        Returns a tensor that equals `tgt` in the forward pass but carries
        gradients with respect to `src`.
        """
        orig_shape = src.shape
        src_flat = src.reshape(-1, src.shape[-1])
        tgt_flat = tgt.reshape(-1, tgt.shape[-1])

        norm_src = src_flat.norm(dim=-1, keepdim=True)
        norm_tgt = tgt_flat.norm(dim=-1, keepdim=True)

        rotated_tgt = VectorQuantizer.rotation_trick(
            VectorQuantizer.safe_div(src_flat, norm_src), VectorQuantizer.safe_div(tgt_flat, norm_tgt), src_flat
        )

        rotated = rotated_tgt * VectorQuantizer.safe_div(norm_tgt, norm_src).detach()
        return rotated.reshape(orig_shape)

    def find_nearest_codes(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Return the indices and code vectors of the nearest codebook entries to `x`."""
        input_shape = x.shape
        flat_x = x.reshape(-1, self.codebook_embedding_dim)

        distances = torch.cdist(flat_x, self.embedding.weight)
        indices = distances.argmin(dim=1)
        quantized = self.embedding(indices).view(input_shape)

        return indices.view(input_shape[:-1]), quantized

    def quantize(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Look up the nearest codebook vectors without computing losses or updating usage stats."""
        return self.find_nearest_codes(x)

    def update_usage(self, indices: Tensor):
        """Increment per-code usage counters by the number of times each code was chosen."""
        indices_flat = indices.flatten()
        self.usage_count.scatter_add_(0, indices_flat, torch.ones_like(indices_flat, dtype=torch.float))
        self.update_count += 1

    def get_usage_rate(self) -> float:
        """Return the fraction of codebook vectors used at least once since the last reset."""
        if self.update_count == 0:
            return 0.0
        return (self.usage_count > 0).float().mean().item()

    def get_max_usage_share(self) -> float:
        """Return the share of all usage claimed by the single most-used code."""
        total = self.usage_count.sum()
        if total == 0:
            return 0.0
        return (self.usage_count.max() / total).item()

    def reset_usage_count(self):
        """Zero out per-code usage counters (e.g. between periodic resets)."""
        self.usage_count.zero_()

    def forward(self, x: Tensor) -> QuantizationOutput:
        """Quantize `x` to the nearest codebook entries, returning losses and STE-propagated output."""
        indices, quantized = self.find_nearest_codes(x)

        commitment_loss = F.mse_loss(quantized.detach(), x)
        codebook_loss = F.mse_loss(quantized, x.detach())
        loss = codebook_loss + self.commitment_weight * commitment_loss

        if self.training:
            quantized_st = VectorQuantizer.rotate_to(x, quantized)
        else:
            quantized_st = x + (quantized - x).detach()

        if self.training:
            self.update_usage(indices)

        logger.debug(
            "VQ forward: batch=%d, codebook_loss=%.4f, commitment_loss=%.4f, loss=%.4f",
            x.shape[0], codebook_loss.item(), commitment_loss.item(), loss.item(),
        )

        return QuantizationOutput(
            quantized_st=quantized_st,
            quantized=quantized,
            indices=indices,
            loss=loss,
            codebook_loss=codebook_loss,
            commitment_loss=commitment_loss,
        )

    def reset_unused_codebook_vectors(self, batch_data: Tensor, dominance_threshold: float = None):
        """Reinit dead or over-dominant codebook entries from current batch vectors."""
        if self.update_count == 0:
            return

        unused_indices = (self.usage_count == 0).nonzero().squeeze(-1)
        dominant_indices = unused_indices.new_empty(0)
        total_usage = self.usage_count.sum()
        if dominance_threshold is not None and dominance_threshold < 1.0 and total_usage > 0:
            dominant_indices = (self.usage_count / total_usage > dominance_threshold).nonzero().squeeze(-1)

        reset_indices = torch.unique(torch.cat([unused_indices, dominant_indices]))

        if len(reset_indices) > 0:
            batch_flat = batch_data.reshape(-1, self.codebook_embedding_dim)
            if batch_flat.shape[0] >= len(reset_indices):
                random_indices = torch.randperm(batch_flat.shape[0], device=batch_flat.device)[: len(reset_indices)]
                self.embedding.weight.data[reset_indices] = batch_flat[random_indices].detach()
                logger.info(
                    "Reset %d/%d codebook vectors from batch samples (%d dead, %d over-dominant)",
                    len(reset_indices), self.codebook_size, len(unused_indices), len(dominant_indices),
                )
            else:
                logger.warning(
                    "Skipped codebook reset: %d codes need resetting but only %d batch samples available",
                    len(reset_indices), batch_flat.shape[0],
                )

        self.reset_usage_count()