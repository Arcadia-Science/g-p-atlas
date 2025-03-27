import json
import pathlib
import pickle
import time
from typing import Literal, cast

import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
from captum.attr import FeatureAblation
from numpy.typing import NDArray
from pydantic import BaseModel
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data.dataset import Dataset

# define a minimum value for variables to avoid divide by zero and buffer underflow.
EPS = 1e-15

LEAKY_RELU_SLOPE = 0.01


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true + EPS), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


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


class GPAtlasTrainerConfig(BaseModel):
    """
    Configuration parameters for a G-P Atlas model trainer.
    """

    # Directory in which to save training logs, checkpoints, and evaluation results.
    output_dirpath: pathlib.Path

    # Directory from which to hot-start the training process.
    hot_start_dirpath: pathlib.Path | None = None

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


class GPAtlasDataset(Dataset):
    """
    A class for importing simulated genotype and phenotype data.

    It expects a pickled object that is organized as a dictionary of arrays:
    'genotypes': array of shape [n_animals, n_loci, n_alleles] (one hot at allelic state)
    'phenotypes': array of shape [n_animals, n_phens] (float values of the phenotypes)
    """

    def __init__(self, filepath: str, num_phenotypes: int):
        """
        Initialize a GPAtlasDataset.

        NOTE: `num_phenotypes` was set to `args.n_phens_to_analyze` in the original implementation.
        """
        with open(filepath, "rb") as f:
            self.dataset = pickle.load(f)

        self.phenotypes = torch.tensor(np.array(self.dataset["phenotypes"]), dtype=torch.float32)
        self.genotypes = torch.tensor(np.array(self.dataset["genotypes"]), dtype=torch.float32)

        if len(self.phenotypes) != len(self.genotypes):
            raise ValueError("The number of phenotypes and genotypes must be the same.")

        # Crop the phenotypes to the specified number.
        # TODO: is it best to do this by taking the first `num_phenotypes` phenotypes?
        self.phenotypes = self.phenotypes[:, :num_phenotypes]

    def __len__(self) -> int:
        return len(self.genotypes)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        genotype = torch.flatten(self.genotypes[idx])
        phenotypes = self.phenotypes[idx]
        return genotype, phenotypes


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
                nn.BatchNorm1d(output_dim, momentum=batchnorm_momentum),
                nn.LeakyReLU(LEAKY_RELU_SLOPE),
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

    def load_from_checkpoint(self, dirpath: str | pathlib.Path) -> None:
        """
        Load weights from a prior run or checkpoint.
        """
        dirpath = pathlib.Path(dirpath)
        for layer_name in ["phenotype_encoder", "phenotype_decoder", "genotype_encoder"]:
            layer = getattr(self, layer_name)
            layer.load_state_dict(
                torch.load(dirpath / f"{layer_name}_state.pt", weights_only=True),
                strict=False,
            )

    def save_to_checkpoint(self, dirpath: str | pathlib.Path) -> None:
        """
        Save the model to a checkpoint.
        """
        dirpath = pathlib.Path(dirpath)
        for layer_name in ["phenotype_encoder", "phenotype_decoder", "genotype_encoder"]:
            layer = getattr(self, layer_name)
            torch.save(layer.state_dict(), dirpath / f"{layer_name}_state.pt")

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


class GPAtlasTrainer:
    """
    A class for training a G-P Atlas model in two stages:
    1. A phenotype-phenotype autoencoder is trained to reconstruct phenotypes.
    2. A genotype encoder is trained to embed genotypes into the latent space
        learned by the phenotype autoencoder by using the phenotype decoder trained
        in the first stage to decode the genotype embeddings back to the phenotype space.
    """

    def __init__(
        self, model: GPAtlasModel, config: GPAtlasTrainerConfig, device: torch.device
    ) -> None:
        """
        Initialize a trainer for a G-P Atlas model.
        """
        self.model = model
        self.config = config
        self.device = device

        self.phenotype_encoder_decoder_training_losses: list[float] = []
        self.genotype_encoder_training_losses: list[float] = []

        torch.manual_seed(self.config.seed)

        self.phenotype_decoder_optimizer = torch.optim.Adam(
            model.phenotype_decoder.parameters(),
            lr=self.config.learning_rate,
            betas=self.config.betas,
        )
        self.phenotype_encoder_optimizer = torch.optim.Adam(
            model.phenotype_encoder.parameters(),
            lr=self.config.learning_rate,
            betas=self.config.betas,
        )
        self.genotype_encoder_optimizer = torch.optim.Adam(
            model.genotype_encoder.parameters(),
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
        # First train the phenotype encoder and decoder.
        self._train_phenotype_encoder_decoder(loader=loader)

        # Freeze the phenotype encoder and decoder before training the genotype encoder.
        self.model.freeze_phenotype_encoder_decoder()

        # With the phenotype encoder and decoder frozen, train the genotype encoder.
        self._train_genotype_encoder(loader=loader)

    def plot_training_losses(self) -> None:
        """
        Plot all of the training losses.
        """
        plt.plot(self.phenotype_encoder_decoder_training_losses, label="Phenotype encoder-decoder")
        plt.plot(self.genotype_encoder_training_losses, label="Genotype encoder")
        plt.title("Training losses")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(self.config.output_dirpath / "reconstruction_loss.svg")
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

        all_original_phenotypes = np.zeros(
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

            all_original_phenotypes[ind] = phenotypes.detach().cpu().numpy()
            all_decoded_phenotypes[ind] = decoded_phenotypes.detach().cpu().numpy()
            all_embeddings[ind] = embeddings.detach().cpu().numpy()

        return (all_original_phenotypes, all_decoded_phenotypes, all_embeddings)

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


def analyze_predictions(
    original_phenotypes: NDArray[np.float32],
    decoded_phenotypes: NDArray[np.float32],
    embeddings: NDArray[np.float32],
    feature_attributions: NDArray[np.float32],
    dataset_path: str | pathlib.Path,
    num_output_phenotypes: int,
    encoder_type: Literal["genotype", "phenotype"] = "genotype",
) -> None:
    """
    Evaluate the performance of each model, saving summaries of model performance.

    Parameters
    ----------
    original_phenotypes : list
        List of real phenotype values (numpy arrays) from model evaluation
    decoded_phenotypes : list
        List of predicted phenotype values (numpy arrays) from model evaluation
    embeddings : list
        List of latent space representations (numpy arrays) from model evaluation
    feature_attributions : list
        List of feature attribution values (numpy arrays) from captum analysis
    dataset_path : str
        Path to directory where all output files should be saved
    num_output_phenotypes : int
        Number of phenotypes to analyze and plot (uses first num_output_phenotypes phenotypes)
    encoder_type : str, optional
        Type of prediction being analyzed - either 'genotype' or 'phenotype'.
        Affects file naming and data saving. Default is 'genotype'

    Returns
    -------
    None
    """
    dataset_path = pathlib.Path(dataset_path)

    # "PE" for "phenotype encoder" and "GE" for "genotype encoder".
    suffix = "PE" if encoder_type == "phenotype" else "GE"

    # Plot and save feature attributions.
    plt.hist(feature_attributions, bins=20)
    plt.savefig(dataset_path / f"{encoder_type}_feature_attributions.svg")
    plt.close()
    with open(dataset_path / f"{encoder_type}_feature_attributions.pk", "wb") as file:
        pickle.dump(feature_attributions, file)

    original_phenotypes = original_phenotypes.T
    decoded_phenotypes = decoded_phenotypes.T
    embeddings = embeddings.T

    # Save the original, decoded, and encoded phenotypes.
    with open(dataset_path / f"phens_phen_encodings_dng_attr{suffix}.pk", "wb") as file:
        pickle.dump(
            [original_phenotypes, decoded_phenotypes, embeddings],
            file,
        )

    # Plot decoded phenotypes vs. original phenotypes.
    for n in range(num_output_phenotypes):
        plt.plot(original_phenotypes[n], decoded_phenotypes[n], "o")

    plt.xlabel("real")
    plt.ylabel("predicted")
    plt.gca().set_aspect("equal")
    plt.savefig(dataset_path / f"phen_real_pred_dng_attr{suffix}.svg")
    plt.close()

    summary_stats: dict[str, list[float]] = {}

    # Pearson correlation.
    pearson_correlations = [
        sc.stats.pearsonr(original_phenotypes[n], decoded_phenotypes[n])[0]
        for n in range(num_output_phenotypes)
    ]
    summary_stats["pearson_correlation"] = cast(list[float], pearson_correlations)
    plt.hist(pearson_correlations, bins=20)
    plt.savefig(dataset_path / f"phen_real_pred_pearson_correlation_dng_attr{suffix}.svg")
    plt.close()

    # Mean squared error.
    mse_values = [
        mean_squared_error(original_phenotypes[n], decoded_phenotypes[n])
        for n in range(num_output_phenotypes)
    ]
    summary_stats["mse"] = cast(list[float], mse_values)
    plt.hist(mse_values, bins=20)
    plt.savefig(dataset_path / f"phen_real_pred_mse_dng_attr{suffix}.svg")
    plt.close()

    # Mean absolute percentage error.
    mape_values = [
        mean_absolute_percentage_error(original_phenotypes[n], decoded_phenotypes[n])
        for n in range(num_output_phenotypes)
    ]
    summary_stats["mape"] = cast(list[float], mape_values)
    plt.hist(mape_values, bins=20)
    plt.savefig(dataset_path / f"phen_real_pred_mape_dng_attr{suffix}.svg")
    plt.close()

    # R-squared
    r2_values = [
        r2_score(original_phenotypes[n], decoded_phenotypes[n])
        for n in range(num_output_phenotypes)
    ]
    summary_stats["r2"] = cast(list[float], r2_values)
    plt.hist(r2_values, bins=20)
    plt.savefig(dataset_path / f"phen_real_pred_r2_dng_attr{suffix}.svg")
    plt.close()

    with open(dataset_path / f"test_stats{suffix}.json", "w") as file:
        json.dump(summary_stats, file)


def main():
    """
    Main function for training and evaluating the G-P Atlas model.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # TODO: load the model and trainer configs from a file.
    model_config = GPAtlasModelConfig()
    trainer_config = GPAtlasTrainerConfig()

    # TODO: add a CLI argument for the dataset filepath.
    dataset_filepath = "/some/path/to/dataset/"

    train_dataset = GPAtlasDataset(
        filepath=(dataset_filepath + "train.pk"), num_phenotypes=model_config.num_input_phenotypes
    )

    test_dataset = GPAtlasDataset(
        filepath=(dataset_filepath + "test.pk"), num_phenotypes=model_config.num_input_phenotypes
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=trainer_config.batch_size,
        num_workers=trainer_config.num_cores,
        shuffle=True,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=1,
        num_workers=trainer_config.num_cores,
        # Important: we do *not* shuffle the test set so that the order of the samples
        # is preserved across evaluation loops.
        shuffle=False,
    )

    model = GPAtlasModel(config=model_config)

    if trainer_config.hot_start:
        model.load_from_checkpoint(dirpath=trainer_config.hot_start_dirpath)

    trainer = GPAtlasTrainer(model=model, config=trainer_config, device=device)

    trainer.train(loader=train_loader)
    trainer.plot_training_losses()

    encoder_types: list[Literal["genotype", "phenotype"]] = ["genotype", "phenotype"]

    for encoder_type in encoder_types:
        # Predict phenotypes.
        (
            all_original_phenotypes,
            all_decoded_phenotypes,
            all_encoded_genotypes,
        ) = trainer.predict_phenotypes(loader=test_loader, encoder_type=encoder_type)

        # Calculate the importance of each input feature for predicting all of the phenotypes.
        feature_attributions = trainer.calculate_feature_importances(
            loader=test_loader, encoder_type=encoder_type
        )

        analyze_predictions(
            original_phenotypes=np.array(all_original_phenotypes),
            decoded_phenotypes=np.array(all_decoded_phenotypes),
            embeddings=np.array(all_encoded_genotypes),
            feature_attributions=feature_attributions,
            dataset_path=trainer_config.output_dirpath,
            num_output_phenotypes=model_config.num_output_phenotypes,
            encoder_type=encoder_type,
        )


if __name__ == "__main__":
    main()
