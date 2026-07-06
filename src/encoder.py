from normalization import L2NormalizationLayer
from typing import List
from torch import nn
from torch import Tensor

class MLP(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: List[int],
            output_dim: int,
            dropout: float = 0.0,
            normalize: bool = False,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout

        dims = [self.input_dim] + self.hidden_dim + [self.output_dim]

        self.mlp = nn.Sequential()
        for i, (input_d, output_d) in enumerate(zip(dims[:-1], dims[1:])):
            self.mlp.append(nn.Linear(input_d, output_d, bias=False))
            if i != len(dims) - 2:
                self.mlp.append(nn.ReLU())
                if dropout != 0:
                    self.mlp.append(nn.Dropout(self.dropout))

        self.mlp.append(L2NormalizationLayer() if normalize else nn.Identity())

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Invalid input dimension: Expected {self.input_dim}, got {x.shape[-1]}"
            )

        return self.mlp(x)