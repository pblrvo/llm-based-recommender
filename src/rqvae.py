from torch import nn
from torch.nn import functional as F
from encoder import MLP
from vector_quantizer import VectorQuantizer
from torch import Tensor
from typing import List, Tuple
from sklearn.cluster import KMeans
import torch

from config import RQVAEConfig

__all__ = ["RQVAEConfig", "RQVAE"]


class RQVAE(nn.Module):

    def __init__(self, config: RQVAEConfig):
        super().__init__()

        self.config = config
        self.item_embedding_dim = config.item_embedding_dim
        self.encoder_hidden_dims = config.encoder_hidden_dims
        self.codebook_embedding_dim = config.codebook_embedding_dim
        self.codebook_quantization_levels = config.codebook_quantization_levels
        self.codebook_normalize = config.codebook_normalize
        self.codebook_size = config.codebook_size

        # Build encoder
        self.encoder = MLP(
            self.item_embedding_dim, self.encoder_hidden_dims, self.codebook_embedding_dim,
            normalize=self.codebook_normalize,
        )

        # Build decoder
        self.decoder = MLP(
            self.codebook_embedding_dim, self.encoder_hidden_dims[::-1], self.item_embedding_dim,
            normalize=False,
        )

        # Quantization Layers
        self.vq_layers = nn.ModuleList([VectorQuantizer(config) for _ in range(self.codebook_quantization_levels)])

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def decode(self, x: Tensor) -> Tensor:
        return self.decoder(x)

    def forward(self, x: Tensor) -> Tuple[Tensor, List[Tensor], dict]:
        """Full forward pass through encoder, quantization, and decoder."""
        z = self.encode(x)

        # Residual quantization
        quantized_out = torch.zeros_like(z)
        residual = z

        all_indices = []
        vq_loss = 0
        codebook_losses = []
        commitment_losses = []

        for vq_layer in self.vq_layers:
            vq_output = vq_layer(residual)  # Quantize current residual
            residual = residual - vq_output.quantized.detach()  # Update residual for next level
            quantized_out = quantized_out + vq_output.quantized_st  # Accumulate quantized vectors
            all_indices.append(vq_output.indices)

            vq_loss = vq_loss + vq_output.loss  # Store indices and accumulate loss
            if vq_output.codebook_loss is not None:  # Track individual loss components
                codebook_losses.append(vq_output.codebook_loss)
            commitment_losses.append(vq_output.commitment_loss)

        x_recon = self.decode(quantized_out)  # Decode
        recon_loss = F.mse_loss(x_recon, x)  # Reconstruction loss
        loss = recon_loss + vq_loss  # Total loss

        loss_dict = {
            "loss": loss,
            "recon_loss": recon_loss,
            "vq_loss": vq_loss,
            "codebook_losses": codebook_losses,  # List of losses per level (empty for EMA)
            "commitment_losses": commitment_losses,  # List of losses per level
            "indices": all_indices,  # Store for metric computation
            "residual": residual,  # Store for residual norm calculation
        }

        return x_recon, all_indices, loss_dict

    def encode_to_semantic_ids(self, x: Tensor) -> Tensor:
        """Extract semantic IDs for input batch."""
        with torch.no_grad():
            z = self.encode(x)
            residual = z
            indices_list = []

            for vq_layer in self.vq_layers:
                indices, quantized = vq_layer.quantize(residual)
                indices_list.append(indices)
                residual = residual - quantized

            # Stack indices from all levels
            semantic_ids = torch.stack(indices_list, dim=-1)
        return semantic_ids

    def decode_from_semantic_ids(self, semantic_ids: Tensor) -> Tensor:
        """Decode from semantic IDs."""
        with torch.no_grad():
            # semantic_ids shape: [batch, codebook_quantization_levels]
            quantized_sum = torch.zeros(semantic_ids.shape[0], self.codebook_embedding_dim, device=semantic_ids.device)

            for level, indices in enumerate(semantic_ids.unbind(dim=-1)):
                codes = self.vq_layers[level].embedding(indices)
                quantized_sum += codes

            return self.decode(quantized_sum)

    def calculate_unique_ids_proportion(self, semantic_ids: Tensor) -> float:
        """Calculate proportion of unique semantic IDs in a batch.

        Args:
            semantic_ids: Tensor of shape [batch_size, codebook_quantization_levels]

        Returns:
            Proportion of items with unique semantic IDs (0 to 1)
        """
        batch_size = semantic_ids.shape[0]
        if batch_size <= 1:
            return 1.0

        # Compare all pairs of semantic IDs
        # Shape: [batch_size, 1, codebook_quantization_levels] == [1, batch_size, codebook_quantization_levels]
        ids_expanded_1 = semantic_ids.unsqueeze(1)  # [B, 1, L]
        ids_expanded_2 = semantic_ids.unsqueeze(0)  # [1, B, L]

        # Check which pairs are identical (all levels match)
        matches = (ids_expanded_1 == ids_expanded_2).all(dim=-1)  # [B, B]

        # Ignore self-matches (the diagonal); a duplicate is a match against
        # ANY other item, whether it appears earlier or later in the batch.
        matches.fill_diagonal_(False)

        has_duplicate = matches.any(dim=1)  # [B]

        # Count unique IDs (those that don't have duplicates)
        n_unique = (~has_duplicate).sum().item()

        return n_unique / batch_size

    def calculate_codebook_usage(self) -> List[float]:
        """Get codebook usage rate for each level.

        Returns:
            List of usage percentages for each quantization level
        """
        return [vq_layer.get_usage_rate() for vq_layer in self.vq_layers]

    def calculate_avg_residual_norm(self, residual: Tensor) -> float:
        """Calculate average residual norm after quantization.

        Args:
            residual: Final residual tensor after all quantization levels

        Returns:
            Average L2 norm of the residual
        """
        return residual.norm(dim=-1).mean().item()

    def kmeans_init(self, data_loader, device):
        """Initialize codebooks using k-means on first batch."""
        # Get first batch
        first_batch = next(iter(data_loader))
        if isinstance(first_batch, (list, tuple)):
            first_batch = first_batch[0]
        first_batch = first_batch.to(device)

        # Encode to latent space
        with torch.no_grad():
            z = self.encode(first_batch)

            # Initialize each level's codebook
            residual = z
            for level, vq_layer in enumerate(self.vq_layers):
                residual_np = residual.cpu().numpy().reshape(-1, self.codebook_embedding_dim)  # Flatten for k-means

                kmeans = KMeans(n_clusters=self.codebook_size, n_init=10, random_state=0)
                kmeans.fit(residual_np)  # Run k-means

                # KMeans always returns float64 centers; cast to match the
                # model's (float32) dtype, or later matmuls/cdist calls fail.
                vq_layer.embedding.weight.data = torch.from_numpy(kmeans.cluster_centers_).float().to(device)

                if level < self.codebook_quantization_levels - 1:  # Compute next residual
                    _, quantized = vq_layer.quantize(residual)
                    residual = residual - quantized
