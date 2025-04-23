import pathlib
import time
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from captum.attr import FeatureAblation
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from pydantic import BaseModel, computed_field
from tqdm import tqdm

from gpatlas.constants import EPS
from gpatlas.models import GPAtlasModel
from gpatlas.utils import get_device

SEPARATOR = " | "


class GPAtlasTrainerConfig(BaseModel):
    """
    Configuration parameters for a G-P Atlas model trainer.
    """

    seed: int = 47

    num_cores: int = 1  # this was originally `args.n_cpu`.

    # TODO: add early-stopping criteria.
    num_epochs: int = 100

    batch_size: int = 16
    learning_rate: float = 0.001  # This was originally `args.lr_r`.
    betas: tuple[float, float] = (0.5, 0.999)  # This was originally `args.b1` and `args.b2`

    phenotype_noise_scale: float = 0.1  # This was originally `args.sd_noise`.

    # This is the probability of randomly flipping a one-hot encoded genotype.
    genotype_noise_scale: float = 0.3  # This was originally `1 - args.gen_noise`.

    l1_lambda: float = 0.8

    @computed_field
    @property
    def num_warmup_epochs(self) -> int:
        """
        The number of warmup epochs before the model starts saving checkpoints.
        """
        return int(self.num_epochs / 3)


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

    def __init__(
        self,
        model: GPAtlasModel,
        config: GPAtlasTrainerConfig,
        checkpoint_filepath: pathlib.Path | None = None,
    ) -> None:
        """
        Initialize a trainer for a G-P Atlas model.
        """
        self.model = model
        self.config = config
        self.checkpoint_filepath = checkpoint_filepath

        torch.manual_seed(self.config.seed)

        self.device = get_device()
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

        best_loss = float("inf")
        training_losses: list[float] = []

        for epoch_idx in range(self.config.num_epochs):
            epoch_start_time = time.time()
            epoch_loss = 0

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

                epoch_loss += float(loss.item())

            epoch_loss /= len(loader)
            training_losses.append(epoch_loss)
            epoch_elapsed_time = time.time() - epoch_start_time

            print(
                SEPARATOR.join(
                    [
                        f"Epoch: {epoch_idx}",
                        f"Training loss: {epoch_loss:.2f}",
                        f"Epoch duration: {epoch_elapsed_time:.2f}s",
                    ]
                )
            )

            if (
                epoch_idx > self.config.num_warmup_epochs
                and epoch_loss < best_loss
                and self.checkpoint_filepath is not None
            ):
                best_loss = epoch_loss
                self.model.save_to_checkpoint(checkpoint_filepath=self.checkpoint_filepath)
                print(f"Saved new best model to {self.checkpoint_filepath}")

        self.phenotype_encoder_decoder_training_losses = training_losses

    def _train_genotype_encoder(
        self, loader: torch.utils.data.DataLoader[tuple[torch.Tensor, torch.Tensor]]
    ) -> None:
        """
        Train the genotype encoder, using the phenotype decoder trained in the first stage.
        """
        self.model.genotype_encoder.train()

        best_loss = float("inf")
        training_losses: list[float] = []

        for epoch_idx in range(self.config.num_epochs):
            epoch_start_time = time.time()
            epoch_loss = 0

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

                # Use L1 regularization on the weights of the first layer of the genetic encoder,
                # as many alleles are likely not to be relevant for predicting the phenotypes.
                l1_norm = torch.sum(torch.abs(self.model.genotype_encoder.layers[0].weight))  # type: ignore
                loss += l1_norm * self.config.l1_lambda

                self.genotype_encoder_optimizer.zero_grad()
                loss.backward()
                self.genotype_encoder_optimizer.step()

                epoch_loss += float(loss.detach())

            epoch_loss /= len(loader)
            training_losses.append(epoch_loss)
            epoch_elapsed_time = time.time() - epoch_start_time

            if (
                epoch_idx > self.config.num_warmup_epochs
                and epoch_loss < best_loss
                and self.checkpoint_filepath is not None
            ):
                best_loss = epoch_loss
                self.model.save_to_checkpoint(checkpoint_filepath=self.checkpoint_filepath)
                print(f"Saved new best model to {self.checkpoint_filepath}")

            print(
                SEPARATOR.join(
                    [
                        f"Epoch: {epoch_idx}",
                        f"Training loss: {epoch_loss:.2f}",
                        f"Epoch duration: {epoch_elapsed_time:.2f}s",
                    ]
                )
            )

        self.genotype_encoder_training_losses = training_losses

    def train(
        self,
        loader: torch.utils.data.DataLoader[tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        """
        Train the model.
        """
        self.model.to(self.device)
        self.model.train()

        # First train the phenotype encoder-decoder.
        print("Training phenotype encoder-decoder...")
        self._train_phenotype_encoder_decoder(loader)

        # Load the best model checkpoint after training the phenotype encoder-decoder.
        if self.checkpoint_filepath is not None:
            self.model.load_from_checkpoint(self.checkpoint_filepath)

        # Freeze the phenotype encoder-decoder before training the genotype encoder.
        print("Freezing phenotype encoder-decoder...")
        self.model.freeze_phenotype_encoder_decoder()

        # With the phenotype encoder-decoder frozen, train the genotype encoder.
        print("Training genotype encoder...")
        self._train_genotype_encoder(loader)

    def plot_training_losses(self, filepath: pathlib.Path) -> None:
        """
        Plot all of the training losses.
        """
        plt.plot(self.phenotype_encoder_decoder_training_losses, label="Phenotype encoder-decoder")
        plt.plot(self.genotype_encoder_training_losses, label="Genotype encoder")
        plt.title("Training losses")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(filepath)
        plt.close()

    def predict_phenotypes(
        self,
        loader: torch.utils.data.DataLoader[tuple[torch.Tensor, torch.Tensor]],
        input_type: Literal["genotype", "phenotype"],
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
        """
        Predict phenotypes using either genotypes or phenotypes as input.
        """
        self.model.to(self.device)
        self.model.eval()

        num_samples = len(loader.dataset)  # type: ignore

        all_true_phenotypes = np.zeros(
            (num_samples, self.model.config.num_input_phenotypes),
            dtype=np.float32,
        )
        all_decoded_phenotypes = np.zeros(
            (num_samples, self.model.config.num_output_phenotypes),
            dtype=np.float32,
        )
        all_embeddings = np.zeros(
            (num_samples, self.model.config.latent_dim),
            dtype=np.float32,
        )

        start_idx = 0
        for genotypes, phenotypes in loader:
            genotypes = genotypes.to(self.device)
            phenotypes = phenotypes.to(self.device)
            batch_size = genotypes.shape[0]

            with torch.no_grad():
                embeddings = (
                    self.model.genotype_encoder(genotypes)
                    if input_type == "genotype"
                    else self.model.phenotype_encoder(phenotypes)
                )
                decoded_phenotypes = self.model.phenotype_decoder(embeddings)

            end_idx = start_idx + batch_size

            all_true_phenotypes[start_idx:end_idx, :] = phenotypes.detach().cpu().numpy()
            all_decoded_phenotypes[start_idx:end_idx, :] = decoded_phenotypes.detach().cpu().numpy()
            all_embeddings[start_idx:end_idx, :] = embeddings.detach().cpu().numpy()

            start_idx = end_idx

        return (all_true_phenotypes, all_decoded_phenotypes, all_embeddings)

    def _calculate_genotype_feature_importances(
        self,
        loader: torch.utils.data.DataLoader[tuple[torch.Tensor, torch.Tensor]],
    ) -> NDArray[np.float32]:
        """
        Calculate the importance of each allelic state in the input genotypes
        for predicting all of the phenotypes.
        """

        self.model.to(self.device)
        self.model.eval()

        def forward_func(genotypes: torch.Tensor, true_phenotypes: torch.Tensor) -> torch.Tensor:
            """
            Forward function for the feature ablation method from Captum.
            """
            predicted_phenotypes = self.model.phenotype_decoder(
                self.model.genotype_encoder(genotypes)
            )

            # TODO: use an unreduced loss function to enable calculating per-phenotype importances.
            return F.mse_loss(
                predicted_phenotypes + EPS,
                true_phenotypes[:, : self.model.config.num_output_phenotypes] + EPS,
            )

        ablator = FeatureAblation(forward_func)

        num_samples = len(loader.dataset)  # type: ignore
        num_input_features = self.model.config.num_loci * self.model.config.num_alleles
        attributions = np.zeros((num_samples, num_input_features), dtype=np.float32)

        start_idx = 0
        for genotypes, phenotypes in tqdm(loader, total=len(loader)):
            genotypes = genotypes.to(self.device)
            phenotypes = phenotypes.to(self.device)
            batch_size = genotypes.shape[0]

            with torch.no_grad():
                attribution = ablator.attribute(genotypes, additional_forward_args=phenotypes)

            end_idx = start_idx + batch_size
            attributions[start_idx:end_idx, :] = attribution.squeeze().detach().cpu().numpy()
            start_idx = end_idx

        return attributions

    def _calculate_phenotype_feature_importances(
        self,
        loader: torch.utils.data.DataLoader[tuple[torch.Tensor, torch.Tensor]],
    ) -> NDArray[np.float32]:
        """
        Calculate the importance of each phenotype for predicting all of the phenotypes.
        """
        self.model.to(self.device)
        self.model.eval()

        def forward_func(phenotypes: torch.Tensor, true_phenotypes: torch.Tensor) -> torch.Tensor:
            """
            Forward function for the feature ablation method from Captum.
            """
            predicted_phenotypes = self.model.phenotype_decoder(
                self.model.phenotype_encoder(phenotypes)
            )
            return F.mse_loss(
                predicted_phenotypes + EPS,
                true_phenotypes[:, : self.model.config.num_output_phenotypes] + EPS,
            )

        ablator = FeatureAblation(forward_func)

        num_samples = len(loader.dataset)  # type: ignore
        num_input_features = self.model.config.num_input_phenotypes
        attributions = np.zeros((num_samples, num_input_features), dtype=np.float32)

        start_idx = 0
        for _, phenotypes in tqdm(loader, total=len(loader)):
            phenotypes = phenotypes.to(self.device)
            batch_size = phenotypes.shape[0]

            with torch.no_grad():
                attribution = ablator.attribute(phenotypes, additional_forward_args=phenotypes)

            end_idx = start_idx + batch_size
            attributions[start_idx:end_idx, :] = attribution.squeeze().detach().cpu().numpy()
            start_idx = end_idx

        return attributions

    def calculate_feature_importances(
        self,
        loader: torch.utils.data.DataLoader[tuple[torch.Tensor, torch.Tensor]],
        input_type: Literal["genotype", "phenotype"],
    ) -> NDArray[np.float32]:
        """
        Calculate the importance of each input feature (genotype or phenotype)
        for predicting all of the phenotypes.
        """
        if input_type == "genotype":
            return self._calculate_genotype_feature_importances(loader=loader)
        else:
            return self._calculate_phenotype_feature_importances(loader=loader)
