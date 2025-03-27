import pathlib
from typing import cast

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

    n_loci_measured: int = 3000
    n_alleles: int = 2

    num_input_phenotypes: int = 30  # This was `args.n_phens_to_analyze`
    num_output_phenotypes: int = 30  # This was `args.n_phens_to_predict`

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


class GPAtlasModel:
    """
    The model architecture used for the initial G-P Atlas publication.
    This model consists of a phenotype encoder-decoder and a genotype encoder,
    with no genotype decoder.
    """

    def __init__(self, config: GPAtlasModelConfig) -> None:
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

        # Note that the input dimension is the number of loci multiplied by the number of alleles.
        self.genotype_encoder = Encoder(
            input_dim=(config.n_loci_measured * config.n_alleles),
            hidden_dim=config.genotype_encoder_hidden_dim,
            latent_dim=config.latent_dim,
            batchnorm_momentum=config.batchnorm_momentum,
        )

    def load_from_checkpoint(self, checkpoint_filepath: str | pathlib.Path) -> None:
        """
        Load weights from a previous run or checkpoint.
        """
        checkpoint = torch.load(checkpoint_filepath)
        self.config = cast(GPAtlasModelConfig, checkpoint["model_config"])
        self.phenotype_encoder.load_state_dict(checkpoint["model_state"]["phenotype_encoder"])
        self.phenotype_decoder.load_state_dict(checkpoint["model_state"]["phenotype_decoder"])
        self.genotype_encoder.load_state_dict(checkpoint["model_state"]["genotype_encoder"])

    def save_to_checkpoint(self, checkpoint_filepath: str | pathlib.Path) -> None:
        """
        Save the model to a checkpoint.
        """
        torch.save(
            {
                "model_config": self.config,
                "model_state": {
                    "phenotype_encoder": self.phenotype_encoder.state_dict(),
                    "phenotype_decoder": self.phenotype_decoder.state_dict(),
                    "genotype_encoder": self.genotype_encoder.state_dict(),
                },
            },
            checkpoint_filepath,
        )

    def to(self, device: torch.device) -> None:
        """
        Move the model to a specific device.
        """
        self.phenotype_encoder.to(device)
        self.phenotype_decoder.to(device)
        self.genotype_encoder.to(device)

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
