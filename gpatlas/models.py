import pathlib

import torch
from pydantic import BaseModel

from gpatlas.layers import Decoder, Encoder


class GPAtlasModelConfig(BaseModel):
    """
    Configuration parameters for a G-P Atlas model.

    This configuration attempts to support some plausible model variants by
    - allowing for separate encoders and decoders for both genotypes and phenotypes.
    - providing a way to define models with or without hidden layers in the encoder and decoder.
    """

    num_loci_measured: int = 3000

    # The number of allelic states at each locus.
    num_alleles: int = 2

    # The number of phenotypes to use as input to the encoder.
    # (This was `args.n_phens_to_analyze` in the original implementation.)
    num_input_phenotypes: int = 30

    # The number of phenotypes to train the phenotype autoencoder to reconstruct.
    # This must be less than or equal to `num_input_phenotypes`.
    # (This was `args.n_phens_to_predict` in the original implementation.)
    num_output_phenotypes: int = 30

    # The dimensions of the hidden layers in the encoder and decoder.
    # If set to `None`, the model will not have any hidden layers
    # in the corresponding encoder or decoder.
    phenotype_encoder_hidden_dim: int | None = 32
    phenotype_decoder_hidden_dim: int | None = 32
    genotype_encoder_hidden_dim: int | None = 32

    # No model variants currently use a genotype decoder, so this is unused.
    # It is included for completeness to allow for future model variants.
    genotype_decoder_hidden_dim: int | None = 32

    # The dimension of the shared latent space.
    latent_dim: int = 32

    batchnorm_momentum: float = 0.8


class GPAtlasModel(torch.nn.Module):
    """
    The model architecture used for the initial G-P Atlas publication.
    This model consists of a phenotype encoder-decoder and a genotype encoder,
    with no genotype decoder.
    """

    def __init__(self, config: GPAtlasModelConfig) -> None:
        super().__init__()

        self.config = config

        self.phenotype_encoder = Encoder(
            input_dim=config.num_input_phenotypes,
            hidden_dim=config.phenotype_encoder_hidden_dim,
            latent_dim=config.latent_dim,
            batchnorm_momentum=config.batchnorm_momentum,
        )
        self.phenotype_decoder = Decoder(
            latent_dim=config.latent_dim,
            hidden_dim=config.phenotype_decoder_hidden_dim,
            output_dim=config.num_output_phenotypes,
            batchnorm_momentum=config.batchnorm_momentum,
        )

        # The input dimension of the genotype encoder is the number of loci multiplied by the
        # number of alleles, because we flatten the 2nd and 3rd dimensions of the genotype tensor.
        self.genotype_encoder = Encoder(
            input_dim=(config.num_loci_measured * config.num_alleles),
            hidden_dim=config.genotype_encoder_hidden_dim,
            latent_dim=config.latent_dim,
            batchnorm_momentum=config.batchnorm_momentum,
        )

    def load_from_checkpoint(self, checkpoint_filepath: str | pathlib.Path) -> None:
        """
        Load weights from a previous run or checkpoint.
        """
        checkpoint = torch.load(checkpoint_filepath)
        self.load_state_dict(checkpoint["model_state"])

    def save_to_checkpoint(self, checkpoint_filepath: str | pathlib.Path) -> None:
        """
        Save the model to a checkpoint.
        """
        torch.save({"model_state": self.state_dict()}, checkpoint_filepath)

    def freeze_phenotype_encoder_decoder(self) -> None:
        """
        Freeze the phenotype encoder and decoder and switch to evaluation mode.
        """
        self.phenotype_encoder.requires_grad_(False)
        self.phenotype_decoder.requires_grad_(False)

        self.phenotype_encoder.eval()
        self.phenotype_decoder.eval()

    def freeze_genotype_encoder(self) -> None:
        """
        Freeze the genotype encoder and switch to evaluation mode.
        """
        self.genotype_encoder.requires_grad_(False)
        self.genotype_encoder.eval()

    def autoencode_phenotypes(self, phenotypes: torch.Tensor) -> torch.Tensor:
        """
        Autoencode phenotypes: phenotypes -> latent space -> decoded phenotypes.
        """
        embeddings = self.phenotype_encoder(phenotypes)
        decoded_phenotypes = self.phenotype_decoder(embeddings)
        return decoded_phenotypes

    def forward(self, genotypes: torch.Tensor) -> torch.Tensor:
        """
        Canonical forward pass of the model: genotypes -> latent space -> phenotypes.
        """
        embeddings = self.genotype_encoder(genotypes)
        decoded_phenotypes = self.phenotype_decoder(embeddings)
        return decoded_phenotypes
