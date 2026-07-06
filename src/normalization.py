from torch import nn
from torch import Tensor
from torch.nn import functional as F

from logger import Logger

logger = Logger.get_logger(__name__)


def l2norm(x, dim=-1, eps=1e-12):
    return F.normalize(x, p=2, dim=dim, eps=eps)

class L2NormalizationLayer(nn.Module):
    def __init__(self, dim=-1, eps=1e-12):
        super().__init__()
        self.dim = dim
        self.eps = eps
        logger.debug("L2NormalizationLayer initialized: dim=%s, eps=%s", dim, eps)

    def forward(self, x) -> Tensor:
        return l2norm(x, dim=self.dim, eps=self.eps)
    

