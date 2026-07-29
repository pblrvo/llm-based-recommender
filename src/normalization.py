"""L2 normalization utility and nn.Module wrapper."""

from torch import nn
from torch import Tensor
from torch.nn import functional as F

from logger import Logger

logger = Logger.get_logger(__name__)


def l2norm(x, dim=-1, eps=1e-12):
    """L2-normalize `x` along `dim` with numerical-stability epsilon `eps`."""
    return F.normalize(x, p=2, dim=dim, eps=eps)


class L2NormalizationLayer(nn.Module):
    """nn.Module wrapper around `l2norm`."""

    def __init__(self, dim=-1, eps=1e-12):
        """Store the normalization axis and epsilon.

        Args:
            dim: Axis to normalize over. Defaults to -1 (the last axis).
            eps: Small constant added to the denominator for numerical stability.
        """
        super().__init__()
        self.dim = dim
        self.eps = eps
        logger.debug("L2NormalizationLayer initialized: dim=%s, eps=%s", dim, eps)

    def forward(self, x) -> Tensor:
        """Apply L2 normalization to `x`."""
        return l2norm(x, dim=self.dim, eps=self.eps)