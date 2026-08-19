"""Residual-Quantized VAE for compressing item embeddings into semantic IDs."""

import time

from torch import nn
from torch.nn import functional as F
from encoder import MLP
from vector_quantizer import VectorQuantizer
from torch import Tensor
from typing import List, Tuple
from sklearn.cluster import KMeans
import torch

from config import RQVAEConfig
from logger import Logger

logger = Logger.get_logger(__name__)

__all__ = ["RQVAEConfig", "RQVAE"]


class RQVAE(nn.Module):
    """Encoder + residual-quantization stack + decoder producing semantic IDs and reconstructions."""

    def __init__(self, config: RQVAEConfig):
        """Build the encoder, decoder, and stack of vector quantizers."""
        super().__init__()

        self.config = config
        self.item_embedding_dim = config.item_embedding_dim
        self.encoder_hidden_dims = config.encoder_hidden_dims
        self.codebook_embedding_dim = config.codebook_embedding_dim
        self.codebook_quantization_levels = config.codebook_quantization_levels
        self.codebook_normalize = config.codebook_normalize
        self.codebook_size = config.codebook_size

        self.encoder = MLP(
            self.item_embedding_dim, self.encoder_hidden_dims, self.codebook_embedding_dim,
            normalize=self.codebook_normalize,
        )

        self.decoder = MLP(
            self.codebook_embedding_dim, self.encoder_hidden_dims[::-1], self.item_embedding_dim,
            normalize=False,
        )

        self.vq_layers = nn.ModuleList([VectorQuantizer(config) for _ in range(self.codebook_quantization_levels)])

        n_params = sum(p.numel() for p in self.parameters())
        logger.info(
            "RQVAE initialized: %d -> %d (levels=%d, codebook_size=%d each), %d total parameters",
            self.item_embedding_dim, self.codebook_embedding_dim, self.codebook_quantization_levels,
            self.codebook_size, n_params,
        )

    def encode(self, x: Tensor) -> Tensor:
        """Run the encoder MLP on `x`."""
        return self.encoder(x)

    def decode(self, x: Tensor) -> Tensor:
        """Run the decoder MLP on `x`."""
        return self.decoder(x)

    def forward(self, x: Tensor) -> Tuple[Tensor, List[Tensor], dict]:
        """Encode, residual-quantize, and decode; return reconstruction, per-level indices, and a loss dict."""
        z = self.encode(x)

        quantized_out = torch.zeros_like(z)
        residual = z

        all_indices = []
        level_residuals = []
        vq_loss = 0
        codebook_losses = []
        commitment_losses = []

        for vq_layer in self.vq_layers:
            level_residuals.append(residual)
            vq_output = vq_layer(residual)
            residual = residual - vq_output.quantized.detach()
            quantized_out = quantized_out + vq_output.quantized_st
            all_indices.append(vq_output.indices)

            vq_loss = vq_loss + vq_output.loss
            if vq_output.codebook_loss is not None:
                codebook_losses.append(vq_output.codebook_loss)
            commitment_losses.append(vq_output.commitment_loss)

        x_recon = self.decode(quantized_out)
        recon_loss = F.mse_loss(x_recon, x)
        loss = recon_loss + vq_loss

        logger.debug(
            "RQVAE forward: batch=%d, loss=%.4f, recon_loss=%.4f, vq_loss=%.4f",
            x.shape[0], loss.item(), recon_loss.item(),
            vq_loss.item() if isinstance(vq_loss, Tensor) else vq_loss,
        )

        loss_dict = {
            "loss": loss,
            "recon_loss": recon_loss,
            "vq_loss": vq_loss,
            "codebook_losses": codebook_losses,
            "commitment_losses": commitment_losses,
            "indices": all_indices,
            "residual": residual,
            "level_residuals": level_residuals,
        }

        return x_recon, all_indices, loss_dict

    def encode_to_semantic_ids(self, x: Tensor) -> Tensor:
        """Encode `x` and return its hierarchical semantic IDs (no gradients), shape [batch, levels]."""
        with torch.no_grad():
            z = self.encode(x)
            residual = z
            indices_list = []

            for vq_layer in self.vq_layers:
                indices, quantized = vq_layer.quantize(residual)
                indices_list.append(indices)
                residual = residual - quantized

            semantic_ids = torch.stack(indices_list, dim=-1)
        logger.info("Encoded %d items to semantic IDs (shape=%s)", x.shape[0], tuple(semantic_ids.shape))
        return semantic_ids

    def decode_from_semantic_ids(self, semantic_ids: Tensor) -> Tensor:
        """Decode a batch of semantic IDs (shape [batch, levels]) back into the original embedding space."""
        with torch.no_grad():
            quantized_sum = torch.zeros(semantic_ids.shape[0], self.codebook_embedding_dim, device=semantic_ids.device)

            for level, indices in enumerate(semantic_ids.unbind(dim=-1)):
                codes = self.vq_layers[level].embedding(indices)
                quantized_sum += codes

            decoded = self.decode(quantized_sum)
        logger.debug("Decoded %d semantic IDs back to embedding space", semantic_ids.shape[0])
        return decoded

    def calculate_unique_ids_proportion(self, semantic_ids: Tensor) -> float:
        """Return the fraction of items in a batch with a unique semantic ID."""
        batch_size = semantic_ids.shape[0]
        if batch_size <= 1:
            return 1.0

        ids_expanded_1 = semantic_ids.unsqueeze(1)
        ids_expanded_2 = semantic_ids.unsqueeze(0)
        matches = (ids_expanded_1 == ids_expanded_2).all(dim=-1)
        matches.fill_diagonal_(False)  # ignore self-matches

        has_duplicate = matches.any(dim=1)
        n_unique = (~has_duplicate).sum().item()

        return n_unique / batch_size

    def calculate_codebook_usage(self) -> List[float]:
        """Return per-level codebook usage rates (fraction of codes used at least once)."""
        return [vq_layer.get_usage_rate() for vq_layer in self.vq_layers]

    def calculate_codebook_max_share(self) -> List[float]:
        """Return per-level single-code usage share (close to 1/codebook_size is healthy; close to 1.0 is collapse)."""
        return [vq_layer.get_max_usage_share() for vq_layer in self.vq_layers]

    def calculate_avg_residual_norm(self, residual: Tensor) -> float:
        """Return the mean L2 norm of the per-item residual after quantization."""
        return residual.norm(dim=-1).mean().item()

    def kmeans_init(self, data_loader, device):
        """Initialize each codebook by running k-means on the first batch's residuals, level by level."""
        first_batch = next(iter(data_loader))
        if isinstance(first_batch, (list, tuple)):
            first_batch = first_batch[0]
        first_batch = first_batch.to(device)

        logger.info(
            "Starting k-means codebook initialization: %d levels, %d clusters/level, %d samples",
            self.codebook_quantization_levels, self.codebook_size, first_batch.shape[0],
        )
        init_start = time.perf_counter()

        with torch.no_grad():
            z = self.encode(first_batch)

            residual = z
            for level, vq_layer in enumerate(self.vq_layers):
                level_start = time.perf_counter()
                residual_np = residual.cpu().numpy().reshape(-1, self.codebook_embedding_dim)

                kmeans = KMeans(n_clusters=self.codebook_size, n_init=10, random_state=0)
                kmeans.fit(residual_np)

                # KMeans returns float64 centers; cast to match the model's float32 dtype.
                vq_layer.embedding.weight.data = torch.from_numpy(kmeans.cluster_centers_).float().to(device)

                logger.info(
                    "K-means init level %d/%d done in %.1fs (inertia=%.4f)",
                    level + 1, self.codebook_quantization_levels, time.perf_counter() - level_start, kmeans.inertia_,
                )

                if level < self.codebook_quantization_levels - 1:
                    _, quantized = vq_layer.quantize(residual)
                    residual = residual - quantized

        logger.info("K-means codebook initialization complete in %.1fs", time.perf_counter() - init_start)