import torch
import torch.nn as nn

from gpatlas.constants import LEAKY_RELU_SLOPE


class Encoder(nn.Module):
    """
    Encoder layer for creating embeddings of genotypes or phenotypes.
    Parameters:
        input_dim: Number of input phenotypes or genotypes.
        hidden_dim: Number of channels in the hidden layer. If `None`, no hidden layer is used.
        latent_dim: Number of dimensions (neurons) in the latent space.
        batchnorm_momentum: Momentum for the batch normalization layers.
    """

    def __init__(
        self, input_dim: int, hidden_dim: int | None, latent_dim: int, batchnorm_momentum: float
    ):
        super().__init__()
        self.layers: nn.Sequential
        if hidden_dim is None:
            self.layers = nn.Sequential(
                nn.Linear(in_features=input_dim, out_features=latent_dim),
                nn.BatchNorm1d(latent_dim, momentum=batchnorm_momentum),
                nn.LeakyReLU(LEAKY_RELU_SLOPE),
            )
        else:
            self.layers = nn.Sequential(
                nn.Linear(in_features=input_dim, out_features=hidden_dim),
                nn.BatchNorm1d(hidden_dim, momentum=batchnorm_momentum),
                nn.LeakyReLU(LEAKY_RELU_SLOPE),
                nn.Linear(in_features=hidden_dim, out_features=latent_dim),
                nn.BatchNorm1d(latent_dim, momentum=batchnorm_momentum),
                nn.LeakyReLU(LEAKY_RELU_SLOPE),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class Decoder(nn.Module):
    """
    Decoder layer for decoding embeddings back to the original data space.
    Parameters:
        latent_dim: Number of dimensions (neurons) in the latent space.
        hidden_dim: Number of channels in the hidden layer. If `None`, no hidden layer is used.
        output_dim: Number of output features.
        batchnorm_momentum: Momentum for the batch normalization layers.
    """

    def __init__(
        self, latent_dim: int, hidden_dim: int | None, output_dim: int, batchnorm_momentum: float
    ):
        super().__init__()
        self.layers: nn.Sequential
        if hidden_dim is None:
            self.layers = nn.Sequential(
                nn.Linear(in_features=latent_dim, out_features=output_dim),
            )
        else:
            self.layers = nn.Sequential(
                nn.Linear(in_features=latent_dim, out_features=hidden_dim),
                nn.BatchNorm1d(hidden_dim, momentum=batchnorm_momentum),
                nn.LeakyReLU(LEAKY_RELU_SLOPE),
                nn.Linear(in_features=hidden_dim, out_features=output_dim),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
