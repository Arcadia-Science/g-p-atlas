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
    learning_rate: float = 0.001  # This was originally `args.lr_r`
    betas: tuple[float, float] = (0.5, 0.999)  # This was originally `args.b1` and `args.b2`

    phenotype_noise_scale: float = 0.1  # This was originally `args.sd_noise`

    # This is the probability of randomly flipping a one-hot encoded genotype.
    genotype_noise_scale: float = 0.3  # This was originally `1 - args.gen_noise`

    L1_lambda: float = 0.8


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

        torch.manual_seed(self.config.seed)

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
        self.model.phenotype_encoder.train()
        self.model.phenotype_decoder.train()

        training_losses: list[float] = []

        for epoch_idx in range(self.config.num_epochs):
            epoch_start_time = time.time()

            for _, phenotypes in loader:
                phenotypes = phenotypes.to(self.device)

                # Add noise to phenotypes.
                noise = torch.randn(phenotypes.shape, device=self.device)
                noisy_phenotypes = phenotypes + (self.config.phenotype_noise_scale**0.5) * noise

                # Encode and decode the phenotypes.
                embeddings = self.model.phenotype_encoder(noisy_phenotypes)
                decoded_phenotypes = self.model.phenotype_decoder(embeddings)

                # Calculate the reconstruction loss.
                loss = F.mse_loss(
                    decoded_phenotypes + EPS,
                    phenotypes[:, : self.model.config.num_output_phenotypes] + EPS,
                )

                # Back propagate and update the model parameters.
                self.phenotype_decoder_optimizer.zero_grad()
                self.phenotype_encoder_optimizer.zero_grad()
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
        Train the genotype encoder, using the phenotype decoder trained in the first stage.
        """
        self.model.genotype_encoder.train()

        training_losses: list[float] = []

        for epoch_idx in range(self.config.num_epochs):
            epoch_start_time = time.time()

            for genotypes, phenotypes in loader:
                genotypes = genotypes.to(self.device)
                phenotypes = phenotypes.to(self.device)

                # Add noise to the genotypes.
                p = self.config.genotype_noise_scale / 2
                pos_noise = torch.bernoulli(torch.full(genotypes.shape, p, device=self.device))
                neg_noise = torch.bernoulli(torch.full(genotypes.shape, p, device=self.device))
                noisy_genotypes = torch.where(genotypes + pos_noise - neg_noise > 0, 1.0, 0.0)

                # Encode and decode the genotypes.
                embeddings = self.model.genotype_encoder(noisy_genotypes)
                decoded_phenotypes = self.model.phenotype_decoder(embeddings)

                loss = F.mse_loss(
                    decoded_phenotypes + EPS,
                    phenotypes[:, : self.model.config.num_output_phenotypes] + EPS,
                )

                # Add L1 regularization to the weights of the first layer of the genetic encoder,
                # as many alleles are likely not to be relevant for predicting the phenotypes.
                L1_norm = torch.linalg.norm(self.model.genotype_encoder.layers[0].weight, ord=1)
                loss += L1_norm * self.config.L1_lambda

                self.genotype_encoder_optimizer.zero_grad()
                loss.backward()
                self.genotype_encoder_optimizer.step()

                training_losses.append(float(loss.detach()))

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
        self.model.to(self.device)
        self.model.train()

        # First train the phenotype encoder-decoder.
        print("Training phenotype encoder-decoder...")
        self._train_phenotype_encoder_decoder(loader=loader)

        # Freeze the phenotype encoder-decoder before training the genotype encoder.
        print("Freezing phenotype encoder-decoder...")
        self.model.freeze_phenotype_encoder_decoder()

        # With the phenotype encoder-decoder frozen, train the genotype encoder.
        print("Training genotype encoder...")
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
        self.model.eval()

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
            genotypes = genotypes.to(self.device)
            phenotypes = phenotypes.to(self.device)

            with torch.no_grad():
                embeddings = (
                    self.model.genotype_encoder(genotypes)
                    if encoder_type == "genotype"
                    else self.model.phenotype_encoder(phenotypes)
                )
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
        self.model.eval()

        if encoder_type not in ["genotype", "phenotype"]:
            raise ValueError(f"Invalid kind: {encoder_type}")

        def forward_func(inputs: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
            """
            Forward function for the feature ablation method from Captum.
            """
            genotypes, phenotypes = inputs
            embeddings = (
                self.model.genotype_encoder(genotypes)
                if encoder_type == "genotype"
                else self.model.phenotype_encoder(phenotypes)
            )
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
