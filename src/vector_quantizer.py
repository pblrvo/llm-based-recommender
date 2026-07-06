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
    quantized_st: Tensor
    quantized: Tensor
    indices: Tensor
    loss: Tensor
    codebook_loss: Optional[Tensor]
    commitment_loss: Tensor


class VectorQuantizer(nn.Module):

    def __init__(self, config: RQVAEConfig):
        super().__init__()
        self.codebook_embedding_dim = config.codebook_embedding_dim
        self.codebook_size = config.codebook_size
        self.commitment_weight = config.commitment_weight

        # Learnable codebook
        self.embedding = nn.Embedding(self.codebook_size, self.codebook_embedding_dim)
        self.embedding.weight.data.uniform_(-1 / self.codebook_size, 1 / self.codebook_size)

        # Track codebook usage
        self.register_buffer("usage_count", torch.zeros(self.codebook_size))
        self.register_buffer("update_count", torch.tensor(0))

        logger.info(
            "VectorQuantizer initialized: codebook_size=%d, codebook_dim=%d, commitment_weight=%.3f",
            self.codebook_size, self.codebook_embedding_dim, self.commitment_weight,
        )

    @staticmethod
    def safe_div(num: Tensor, den: Tensor, eps: float = 1e-6) -> Tensor:
        # Safe division to avoid numerical issues
        return num / den.clamp(min=eps)

    @staticmethod
    def rotation_trick(u: Tensor, q: Tensor, e: Tensor) -> Tensor:
        """
        Efficient rotation trick transform from Eq 4.2 in https://arxiv.org/abs/2410.06424

        Args:
            u: Unit vector from encoder output (normalized x)
            q: Unit vector from quantized output (normalized quantized)
            e: Original encoder output (x)

        Returns:
            Rotated encoder output
        """
        w = l2norm(u + q, dim=-1, eps=1e-6).detach()

        # Reshape for batch matrix multiplication
        w_col = w.unsqueeze(-1)
        w_row = w.unsqueeze(-2)
        u_col = u.unsqueeze(-1).detach()
        q_row = q.unsqueeze(-2).detach()

        # For 2D input, add temporary batch dimension
        if e.ndim == 2:
            e_expanded = e.unsqueeze(1)  # [B, D] -> [B, 1, D]
            result = e_expanded - 2 * (e_expanded @ w_col @ w_row) + 2 * (e_expanded @ u_col @ q_row)
            return result.squeeze(1)  # [B, 1, D] -> [B, D]
        else:
            return e - 2 * (e @ w_col @ w_row).squeeze(-1) + 2 * (e @ u_col @ q_row).squeeze(-1)

    @staticmethod
    def rotate_to(src: Tensor, tgt: Tensor) -> Tensor:
        """
        Apply rotation trick STE from https://arxiv.org/abs/2410.06424
        to get gradients through VQ layer.

        Args:
            src: Source tensor (encoder output)
            tgt: Target tensor (quantized output)

        Returns:
            Rotated tensor that equals tgt in forward pass but has gradients
        """
        # Flatten to 2D for processing
        orig_shape = src.shape
        src_flat = src.reshape(-1, src.shape[-1])
        tgt_flat = tgt.reshape(-1, tgt.shape[-1])

        # Get norms
        norm_src = src_flat.norm(dim=-1, keepdim=True)
        norm_tgt = tgt_flat.norm(dim=-1, keepdim=True)

        # Apply rotation in normalized space
        rotated_tgt = VectorQuantizer.rotation_trick(
            VectorQuantizer.safe_div(src_flat, norm_src), VectorQuantizer.safe_div(tgt_flat, norm_tgt), src_flat
        )

        # Scale to match target norm
        rotated = rotated_tgt * VectorQuantizer.safe_div(norm_tgt, norm_src).detach()

        # Reshape back
        return rotated.reshape(orig_shape)

    def find_nearest_codes(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        input_shape = x.shape
        flat_x = x.reshape(-1, self.codebook_embedding_dim)

        # Calculate distances to all codebook vectors
        distances = torch.cdist(flat_x, self.embedding.weight)
        indices = distances.argmin(dim=1)
        quantized = self.embedding(indices).view(input_shape)

        return indices.view(input_shape[:-1]), quantized

    def quantize(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Look up the nearest codebook vectors without computing losses or
        updating usage stats. Use for inference/initialization paths (see
        forward() for the training path, which tracks usage and returns
        gradients via the straight-through estimator)."""
        return self.find_nearest_codes(x)

    def update_usage(self, indices: Tensor):
        """Update codebook usage statistics."""
        indices_flat = indices.flatten()
        self.usage_count.scatter_add_(0, indices_flat, torch.ones_like(indices_flat, dtype=torch.float))
        self.update_count += 1

    def get_usage_rate(self) -> float:
        """Get proportion of codebook vectors that have been used."""
        if self.update_count == 0:
            return 0.0
        return (self.usage_count > 0).float().mean().item()

    def reset_usage_count(self):
        """Reset usage count (useful for periodic resets)."""
        self.usage_count.zero_()

    def forward(self, x: Tensor) -> QuantizationOutput:
        # Find nearest codebook vectors
        indices, quantized = self.find_nearest_codes(x)

        # Compute losses
        commitment_loss = F.mse_loss(quantized.detach(), x)
        codebook_loss = F.mse_loss(quantized, x.detach())
        loss = codebook_loss + self.commitment_weight * commitment_loss

        # Straight-through estimator for gradients
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

    def reset_unused_codebook_vectors(self, batch_data: Tensor):
        """Reset unused codebook vectors to random values."""
        if self.update_count == 0:
            return

        # Find codes with zero usage
        unused_indices = (self.usage_count == 0).nonzero().squeeze(-1)

        if len(unused_indices) > 0:
            batch_flat = batch_data.reshape(-1, self.codebook_embedding_dim)
            if batch_flat.shape[0] >= len(unused_indices):
                # Sample random vectors from batch
                random_indices = torch.randperm(batch_flat.shape[0], device=batch_flat.device)[: len(unused_indices)]
                self.embedding.weight.data[unused_indices] = batch_flat[random_indices].detach()
                logger.info(
                    "Reset %d/%d unused codebook vectors from batch samples",
                    len(unused_indices), self.codebook_size,
                )
            else:
                logger.warning(
                    "Skipped codebook reset: %d unused codes but only %d batch samples available",
                    len(unused_indices), batch_flat.shape[0],
                )

        # Reset usage count after replacement
        self.reset_usage_count()
