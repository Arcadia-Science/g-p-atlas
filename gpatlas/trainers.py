import pathlib
import time
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from captum.attr import FeatureAblation
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from pydantic import BaseModel

from gpatlas.constants import EPS
from gpatlas.models import GPAtlasModel


class GPAtlasTrainerConfig(BaseModel):
    """
    Configuration parameters for a G-P Atlas model trainer.
    """

    seed: int = 47

    num_cores: int = 14  # this was `args.n_cpu`

    # TODO: do we need separate num_epochs for phenotype encoder-decoder and genotype encoder?
    num_epochs: int = 100

    batch_size: int = 16
    learning_rate: float = 0.001  # This was `args.lr_r`
    betas: tuple[float, float] = (0.5, 0.999)  # This was `args.b1` and `args.b2`

    phenotype_noise_scale: float = 0.1  # This was args.sd_noise
    genotype_noise_scale: float = 0.3  # This was args.gen_noise

    L1_lambda: float = 0.8
    L2_lambda: float = 0.01


class GPAtlasTrainer:
    """
    A class for training and evaluating a G-P Atlas model.

    Training is done in two stages:
    1. A phenotype-phenotype autoencoder is trained to reconstruct phenotypes.
    2. A genotype encoder is trained to embed genotypes into the latent space
        learned by the phenotype autoencoder by using the phenotype decoder trained
        in the first stage to decode the genotype embeddings back to the phenotype space.

    Evaluation is done by predicting phenotypes from either genotypes or phenotypes,
    and calculating feature importances for both genotypes and phenotypes.
    """

    def __init__(self, model: GPAtlasModel, config: GPAtlasTrainerConfig) -> None:
        """
        Initialize a trainer for a G-P Atlas model.
        """
        self.model = model
        self.config = config

        # TODO: consider specifying the device to use in the trainer config.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.phenotype_encoder_decoder_training_losses: list[float] = []
        self.genotype_encoder_training_losses: list[float] = []

        self._initialize_optimizers()

    def _initialize_optimizers(self) -> None:
        self.phenotype_decoder_optimizer = torch.optim.Adam(
            self.model.phenotype_decoder.parameters(),
            lr=self.config.learning_rate,
            betas=self.config.betas,
        )
        self.phenotype_encoder_optimizer = torch.optim.Adam(
            self.model.phenotype_encoder.parameters(),
            lr=self.config.learning_rate,
            betas=self.config.betas,
        )
        self.genotype_encoder_optimizer = torch.optim.Adam(
            self.model.genotype_encoder.parameters(),
            lr=self.config.learning_rate,
            betas=self.config.betas,
        )

    def _train_phenotype_encoder_decoder(
        self, loader: torch.utils.data.DataLoader[tuple[torch.Tensor, torch.Tensor]]
    ) -> None:
        """
        Train the phenotype encoder and decoder.
        """

        self.model.to(self.device)
        training_losses: list[float] = []

        for epoch_idx in range(self.config.num_epochs):
            epoch_start_time = time.time()

            for _, phenotypes in enumerate(loader):
                phenotypes = phenotypes.to(self.device)

                # Initialize gradients for training.
                self.model.phenotype_decoder.zero_grad()
                self.model.phenotype_encoder.zero_grad()

                # Add noise to phenotypes.
                noise = torch.randn(phenotypes.shape, device=self.device)
                noisy_phenotypes = phenotypes + (self.config.phenotype_noise_scale**0.5) * noise

                # Encode and decode the phenotypes.
                embeddings = self.model.phenotype_encoder(noisy_phenotypes)
                decoded_phenotypes = self.model.phenotype_decoder(embeddings)

                # Calculate the reconstruction loss.
                # TODO: `num_output_phenotypes` will only have an effect here
                # if it is less than `num_input_phenotypes`.
                loss = F.mse_loss(
                    decoded_phenotypes + EPS,
                    phenotypes[:, : self.model.config.num_output_phenotypes] + EPS,
                )

                # Back propagate and update the model parameters.
                loss.backward()
                self.phenotype_decoder_optimizer.step()
                self.phenotype_encoder_optimizer.step()

                training_losses.append(float(loss.detach()))

            elapsed_time = time.time() - epoch_start_time
            print(
                f"Epoch: {epoch_idx}, "
                f"Training loss: {training_losses[-1]}, "
                f"Epoch duration: {elapsed_time}"
            )

        self.phenotype_encoder_decoder_training_losses = training_losses

    def _train_genotype_encoder(
        self, loader: torch.utils.data.DataLoader[tuple[torch.Tensor, torch.Tensor]]
    ) -> None:
        """
        Train the genotype encoder.
        This assumes that the phenotype encoder and decoder are pre-trained and frozen.
        """
        self.model.to(self.device)
        training_losses: list[float] = []

        for epoch_idx in range(self.config.num_epochs):
            epoch_start_time = time.time()

            for genotypes, phenotypes in loader:
                genotypes = genotypes.to(self.device)
                phenotypes = phenotypes.to(self.device)

                # TODO: is this indexing necessary?
                # Can we instead flatten the 2nd and 3rd dimensions of `genotypes`?
                genotypes = genotypes[
                    :, : self.model.config.n_loci_measured * self.model.config.n_alleles
                ]

                # TODO: refactor to use torch.bernoulli in order to ensure reproducibility here.
                # (because torch.manual_seed won't affect np.random.binomial).
                pos_noise = np.random.binomial(
                    1, (1 - self.config.genotype_noise_scale) / 2, genotypes.shape
                )
                neg_noise = np.random.binomial(
                    1, (1 - self.config.genotype_noise_scale) / 2, genotypes.shape
                )

                # Add the noise to the genetic data.
                noisy_genotypes = torch.tensor(
                    np.where((genotypes + pos_noise - neg_noise) > 0, 1, 0),
                    dtype=torch.float32,
                    device=self.device,
                )

                self.model.genotype_encoder.zero_grad()
                embeddings = self.model.genotype_encoder(noisy_genotypes)
                decoded_phenotypes = self.model.phenotype_decoder(embeddings)

                loss = F.mse_loss(
                    decoded_phenotypes + EPS,
                    phenotypes[:, : self.model.config.num_output_phenotypes] + EPS,
                )

                training_losses.append(float(loss.detach()))

                # Calculate the L1 and L2 norms for the weights in the first layer
                # of the genetic encoder and add them to the reconstruction loss.
                # TODO: why do we take a sum over the first dimension here?
                # This tensor should have shape `(latent_dim, n_loci * n_alleles)`.
                weights_sum = torch.sum(self.model.genotype_encoder.layers[0].weight, axis=0)  # type: ignore

                L1_norm = torch.linalg.norm(weights_sum, 1)
                L2_norm = torch.linalg.norm(weights_sum, 2)

                # TODO: shouldn't the L2 norm be squared here?
                loss += L1_norm * self.config.L1_lambda + L2_norm * self.config.L2_lambda

                loss.backward()
                self.genotype_encoder_optimizer.step()

            elapsed_time = time.time() - epoch_start_time
            print(
                f"Epoch: {epoch_idx}, "
                f"Training loss: {training_losses[-1]}, "
                f"Epoch duration: {elapsed_time}"
            )

        self.genotype_encoder_training_losses = training_losses

    def train(self, loader: torch.utils.data.DataLoader[tuple[torch.Tensor, torch.Tensor]]) -> None:
        """
        Train the model.
        """
        torch.manual_seed(self.config.seed)
        self.model.to(self.device)

        # First train the phenotype encoder and decoder.
        self._train_phenotype_encoder_decoder(loader=loader)

        # Freeze the phenotype encoder and decoder before training the genotype encoder.
        self.model.freeze_phenotype_encoder_decoder()

        # With the phenotype encoder and decoder frozen, train the genotype encoder.
        self._train_genotype_encoder(loader=loader)

    def plot_training_losses(self, output_dirpath: pathlib.Path) -> None:
        """
        Plot all of the training losses.
        """
        plt.plot(self.phenotype_encoder_decoder_training_losses, label="Phenotype encoder-decoder")
        plt.plot(self.genotype_encoder_training_losses, label="Genotype encoder")
        plt.title("Training losses")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(output_dirpath / "reconstruction_loss.svg")
        plt.close()

    def predict_phenotypes(
        self,
        loader: torch.utils.data.DataLoader[tuple[torch.Tensor, torch.Tensor]],
        encoder_type: Literal["genotype", "phenotype"],
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
        """
        Predict phenotypes using either genotypes or phenotypes as input.
        """
        self.model.to(self.device)

        all_true_phenotypes = np.zeros(
            (len(loader), self.model.config.num_input_phenotypes),
            dtype=np.float32,
        )
        all_decoded_phenotypes = np.zeros(
            (len(loader), self.model.config.num_output_phenotypes),
            dtype=np.float32,
        )
        all_embeddings = np.zeros(
            (len(loader), self.model.config.latent_dim),
            dtype=np.float32,
        )

        for ind, (genotypes, phenotypes) in enumerate(loader):
            genotypes = genotypes[
                :, : self.model.config.n_loci_measured * self.model.config.n_alleles
            ]
            genotypes = genotypes.to(self.device)
            phenotypes = phenotypes.to(self.device)

            with torch.no_grad():
                embeddings: torch.Tensor
                if encoder_type == "genotype":
                    embeddings = self.model.genotype_encoder(genotypes)
                elif encoder_type == "phenotype":
                    embeddings = self.model.phenotype_encoder(phenotypes)

                decoded_phenotypes = self.model.phenotype_decoder(embeddings)

            all_true_phenotypes[ind] = phenotypes.detach().cpu().numpy()
            all_decoded_phenotypes[ind] = decoded_phenotypes.detach().cpu().numpy()
            all_embeddings[ind] = embeddings.detach().cpu().numpy()

        return (all_true_phenotypes, all_decoded_phenotypes, all_embeddings)

    def calculate_feature_importances(
        self,
        loader: torch.utils.data.DataLoader[tuple[torch.Tensor, torch.Tensor]],
        encoder_type: Literal["genotype", "phenotype"],
    ) -> NDArray[np.float32]:
        """
        Calculate the importance of each input feature (genotype or phenotype)
        for predicting all of the phenotypes.
        """
        self.model.to(self.device)

        if encoder_type not in ["genotype", "phenotype"]:
            raise ValueError(f"Invalid kind: {encoder_type}")

        def forward_func(inputs: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
            """
            Forward function for the feature ablation method from Captum.
            """
            genotypes, phenotypes = inputs
            embeddings: torch.Tensor
            if encoder_type == "genotype":
                embeddings = self.model.genotype_encoder(genotypes)
            elif encoder_type == "phenotype":
                embeddings = self.model.phenotype_encoder(phenotypes)
            decoded_phenotypes = self.model.phenotype_decoder(embeddings)
            return F.mse_loss(
                decoded_phenotypes + EPS,
                phenotypes[:, : self.model.config.num_output_phenotypes] + EPS,
            )

        ablator = FeatureAblation(forward_func)

        attributions = np.zeros(
            (len(loader), self.model.config.num_output_phenotypes),
            dtype=np.float32,
        )
        for ind, (genotypes, phenotypes) in enumerate(loader):
            genotypes = genotypes.to(self.device)
            phenotypes = phenotypes.to(self.device)
            with torch.no_grad():
                attribution = ablator.attribute(inputs=(genotypes, phenotypes))
            attributions[ind] = attribution[0].squeeze().detach().cpu().numpy()

        return attributions
