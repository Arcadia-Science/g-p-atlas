import pathlib
import pickle
import time
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
from captum.attr import FeatureAblation
from pydantic import BaseModel
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data.dataset import Dataset

# define a minimum value for variables to avoid divide by zero and buffer underflow.
EPS = 1e-15

LEAKY_RELU_SLOPE = 0.01


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true + EPS), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--n_alleles",
        type=int,
        default=2,
        help="number of segregating causal alleles at any given causal locus",
    )
    parser.add_argument(
        "--n_locs",
        type=int,
        default=900,
        help="number of causal loci to model.  This is the same as the number of genes",
    )
    parser.add_argument(
        "--n_env", type=int, default=3, help="number of influential continuous components"
    )
    parser.add_argument("--n_phens", type=int, default=30, help="number of phenotypes")
    parser.add_argument(
        "--gen_lw", type=float, default=1, help="weight for the loss attributed to genetic features"
    )
    parser.add_argument(
        "--eng_lw", type=float, default=0.1, help="weight for the loss attributed to env features"
    )
    parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--lr_r", type=float, default=0.001, help="reconstruction learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: gradient decay variables")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: gradient decay variables")
    parser.add_argument("--n_cpu", type=int, default=14, help="number of cpus")
    parser.add_argument(
        "--e_hidden_dim",
        type=int,
        default=32,
        help="number of neurons in the hidden layers of encoder",
    )
    parser.add_argument(
        "--d_hidden_dim",
        type=int,
        default=32,
        help="number of neurons in the hidden layers of decoder",
    )
    parser.add_argument(
        "--ge_hidden_dim",
        type=int,
        default=32,
        help="number of neurons in the hidden layers of the genetic encoder",
    )
    parser.add_argument(
        "--batchnorm_momentum",
        type=float,
        default=0.8,
        help="momentum for the batch normalization layers",
    )
    parser.add_argument(
        "--latent_dim", type=int, default=32, help="number of neurons in the latent space"
    )
    parser.add_argument(
        "--n_phens_to_analyze", type=int, default=30, help="number of phenotypes to analyze"
    )
    parser.add_argument("--sd_noise", type=float, default=0.1, help="noise added to phens")
    parser.add_argument("--gen_noise", type=float, default=0.3, help="noise added to gens")
    parser.add_argument(
        "--n_phens_to_predict", type=int, default=30, help="number of phenotypes to predict"
    )
    parser.add_argument(
        "--n_epochs_gen",
        type=int,
        default=100,
        help="number of epochs to train the genetic encoder",
    )
    parser.add_argument(
        "--n_loci_measured",
        type=int,
        default=3000,
        help="number of genetic loci for which there is data",
    )
    parser.add_argument(
        "--l1_lambda",
        type=float,
        default=0.8,
        help="l1 regularization weight for the genetic weights",
    )
    parser.add_argument(
        "--l2_lambda",
        type=float,
        default=0.01,
        help="l2 regularization weight for the genetic weights",
    )
    parser.add_argument(
        "--dataset_path", type=str, default=None, help="where the train and test files are located"
    )
    parser.add_argument(
        "--train_suffix",
        type=str,
        default="train.pk",
        help="name of the training data file, defaults to DGRP dataset",
    )
    parser.add_argument(
        "--test_suffix", type=str, default="test.pk", help="name of the test data file"
    )
    parser.add_argument(
        "--hot_start",
        type=bool,
        default=False,
        help="flag to use precomputed weights, false by default",
    )
    parser.add_argument(
        "--hot_start_path_e",
        type=str,
        default=None,
        help="path to the phenotype encoder weights to use at initialization",
    )
    parser.add_argument(
        "--hot_start_path_d",
        type=str,
        default=None,
        help="path to the phenotype decoder weights to use at initialization",
    )
    parser.add_argument(
        "--hot_start_path_ge",
        type=str,
        default=None,
        help="path to the genotype encoder weights to use at initialization",
    )
    parser.add_argument(
        "--calculate_importance",
        type=str,
        default="no",
        help="flag whether to calculate variable importance. Expects 'no' or 'yes.' ",
    )

    return parser.parse_args()


class GPAtlasModelConfig(BaseModel):
    """
    Configuration parameters for a G-P Atlas model.
    """

    n_loci_measured: int = 3000
    n_alleles: int = 2

    num_input_phenotypes: int = 30  # This was `args.n_phens_to_analyze`
    num_output_phenotypes: int = 30  # This was `args.n_phens_to_predict`

    phenotype_encoder_hidden_dim: int = 32
    phenotype_decoder_hidden_dim: int = 32
    genotype_encoder_hidden_dim: int = 32

    latent_dim: int = 32

    batchnorm_momentum: float = 0.8


class GPAtlasTrainerConfig(BaseModel):
    """
    Configuration parameters for a G-P Atlas model trainer.
    """

    seed: int = 47

    num_cores: int = 14
    num_epochs: int = 100
    batch_size: int = 16

    learning_rate: float = 0.001
    betas: tuple[float, float] = (0.5, 0.999)

    phenotype_noise_sd: float = 0.1  # This was args.sd_noise
    genotype_noise_sd: float = 0.3  # This was args.gen_noise

    L1_lambda: float = 0.8
    L2_lambda: float = 0.01


class GPAtlasDataset(Dataset):
    """
    A class for importing simulated genotype and phenotype data.

    It expects a pickled object that is organized as a dictionary of tensors:
    'genotypes': array of shape [n_animals, n_loci, n_alleles] (one hot at allelic state)
    'phenotypes': array of shape [n_animals, n_phens] (float values of the phenotypes)
    """

    def __init__(self, filepath: str, num_phenotypes: int):
        """
        Initialize a GPAtlasDataset.

        NOTE: `n_phens` was set to `n_phens_to_analyze` in the original implementation.
        """
        self.filepath = filepath
        self.num_phenotypes = num_phenotypes

        with open(filepath, "rb") as f:
            self.dataset = pickle.load(f)

        self.phenotypes = torch.tensor(np.array(self.dataset["phenotypes"]), dtype=torch.float32)
        self.genotypes = torch.tensor(np.array(self.dataset["genotypes"]), dtype=torch.float32)

        if len(self.phenotypes) != len(self.genotypes):
            raise ValueError("The number of phenotypes and genotypes must be the same.")

        # Crop the phenotypes to the number of phenotypes to analyze.
        # TODO: is this correct?
        self.phenotypes = self.phenotypes[:, : self.num_phenotypes]

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
        input_dim (int): Number of input phenotypes or genotypes.
        hidden_dim (int): Number of channels in the hidden layer.
        latent_dim (int): Number of dimensions (neurons) in the latent space.
        batchnorm_momentum (float): Momentum for the batch normalization layers.
    """

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, batchnorm_momentum: float):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=hidden_dim),
            nn.BatchNorm1d(hidden_dim, momentum=batchnorm_momentum),
            nn.LeakyReLU(LEAKY_RELU_SLOPE),
            nn.Linear(in_features=hidden_dim, out_features=latent_dim),
            nn.BatchNorm1d(latent_dim, momentum=batchnorm_momentum),
            nn.LeakyReLU(LEAKY_RELU_SLOPE),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class Decoder(nn.Module):
    """
    Decoder layer for decoding embeddings back to the original data space.
    Parameters:
        latent_dim (int): Number of dimensions (neurons) in the latent space.
        hidden_dim (int): Number of channels in the hidden layer.
        output_dim (int): Number of output features.
        batchnorm_momentum (float): Momentum for the batch normalization layers.
    """

    def __init__(
        self, latent_dim: int, hidden_dim: int, output_dim: int, batchnorm_momentum: float
    ):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=hidden_dim),
            nn.BatchNorm1d(hidden_dim, momentum=batchnorm_momentum),
            nn.LeakyReLU(LEAKY_RELU_SLOPE),
            nn.Linear(in_features=hidden_dim, out_features=output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)


class GPAtlasModel:
    def __init__(self, config: GPAtlasModelConfig) -> None:
        """
        Initialize a G-P Atlas model.
        """
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

        # This is for feature importance using captum at evaluation time.
        self.genotype_feature_ablator = FeatureAblation(self.genotype_phenotype_forward)
        self.phenotype_feature_ablator = FeatureAblation(self.phenotype_phenotype_forward)

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

    def genotype_phenotype_forward(
        self, inputs: torch.Tensor, baselines: torch.Tensor
    ) -> torch.Tensor:
        """
        Function to use for calculating feature importance for genotype-phenotype prediction
        using captum's `FeatureAblation` class.
        """
        encoded_genotype = self.genotype_encoder(inputs)
        decoded_phenotype = self.phenotype_decoder(encoded_genotype)
        return F.mse_loss(
            decoded_phenotype + EPS, baselines[:, : self.config.num_output_phenotypes] + EPS
        )

    def phenotype_phenotype_forward(
        self, inputs: torch.Tensor, baselines: torch.Tensor
    ) -> torch.Tensor:
        """
        Function to use for calculating feature importance for phenotype-phenotype prediction
        using captum's `FeatureAblation` class.
        """
        encoded_phenotype = self.phenotype_encoder(inputs)
        decoded_phenotype = self.phenotype_decoder(encoded_phenotype)
        return F.mse_loss(
            decoded_phenotype + EPS, baselines[:, : self.config.num_output_phenotypes] + EPS
        )

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
    def __init__(
        self,
        model: GPAtlasModel,
        config: GPAtlasTrainerConfig,
        train_loader: torch.utils.data.DataLoader[tuple[torch.Tensor, torch.Tensor]],
        device: torch.device,
    ) -> None:
        """
        Initialize a trainer for a G-P Atlas model.
        """
        self.model = model
        self.config = config
        self.train_loader = train_loader
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

    def train_phenotype_encoder_decoder(self) -> None:
        """
        Train the phenotype encoder and decoder.
        """

        self.model.to(self.device)
        training_losses: list[float] = []

        for epoch_idx in range(self.config.num_epochs):
            epoch_start_time = time.time()

            for batch_idx, (_, phenotypes) in enumerate(self.train_loader):
                phenotypes = phenotypes.to(self.device)

                # Initialize gradients for training.
                self.model.phenotype_decoder.zero_grad()
                self.model.phenotype_encoder.zero_grad()

                # Add noise to phenotypes.
                noise = torch.randn(phenotypes.shape, device=self.device)
                noisy_phenotypes = phenotypes + (self.config.phenotype_noise_sd**0.5) * noise

                # Encode and decode the phenotypes.
                encoded_phenotypes = self.model.phenotype_encoder(noisy_phenotypes)
                decoded_phenotypes = self.model.phenotype_decoder(encoded_phenotypes)

                # Calculate the reconstruction loss.
                # TODO: `num_output_phenotypes` will only have an effect here
                # if it is less than `n_phens`.
                epoch_recon_loss = F.mse_loss(
                    decoded_phenotypes + EPS,
                    phenotypes[:, : self.model.config.num_output_phenotypes] + EPS,
                )

                # Back propagate and update the model parameters.
                epoch_recon_loss.backward()
                self.phenotype_decoder_optimizer.step()
                self.phenotype_encoder_optimizer.step()

                training_losses.append(float(epoch_recon_loss.detach()))

            elapsed_time = time.time() - epoch_start_time
            print(
                f"Epoch: {epoch_idx}, batch: {batch_idx}, "
                f"Training loss: {training_losses[-1]}, "
                f"Epoch duration: {elapsed_time}"
            )

        self.phenotype_encoder_decoder_training_losses = training_losses

    def train_genotype_encoder(self) -> None:
        """
        Train the genotype encoder.
        This assumes that the phenotype encoder and decoder are pre-trained and frozen.
        """
        self.model.to(self.device)
        training_losses: list[float] = []

        for epoch_idx in range(self.config.num_epochs):
            epoch_start_time = time.time()

            for batch_idx, (genotypes, phenotypes) in enumerate(self.train_loader):
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
                    1, (1 - self.config.genotype_noise_sd) / 2, genotypes.shape
                )
                neg_noise = np.random.binomial(
                    1, (1 - self.config.genotype_noise_sd) / 2, genotypes.shape
                )

                # Add the noise to the genetic data.
                noisy_genotypes = torch.tensor(
                    np.where((genotypes + pos_noise - neg_noise) > 0, 1, 0),
                    dtype=torch.float32,
                    device=self.device,
                )

                self.model.genotype_encoder.zero_grad()
                encoded_genotypes = self.model.genotype_encoder(noisy_genotypes)
                decoded_phenotypes = self.model.phenotype_decoder(encoded_genotypes)

                batch_recon_loss = F.mse_loss(
                    decoded_phenotypes + EPS,
                    phenotypes[:, : self.model.config.num_output_phenotypes] + EPS,
                )

                training_losses.append(float(batch_recon_loss.detach()))

                # Calculate the L1 and L2 norms for the weights in the first layer
                # of the genetic encoder and add them to the reconstruction loss.
                # TODO: why do we take a sum here prior to taking the norm?
                weights_sum = torch.sum(self.model.genotype_encoder.encoder[0].weight, axis=0)  # type: ignore

                L1_norm = torch.linalg.norm(weights_sum, 1)
                L2_norm = torch.linalg.norm(weights_sum, 2)

                # TODO: shouldn't the L2 norm be squared here?
                batch_recon_loss += (
                    L1_norm * self.config.L1_lambda + L2_norm * self.config.L2_lambda
                )

                batch_recon_loss.backward()
                self.genotype_encoder_optimizer.step()

            elapsed_time = time.time() - epoch_start_time
            print(
                f"Epoch: {epoch_idx}, Batch: {batch_idx}, Training loss: {training_losses[-1]}, "
                f"Epoch duration: {elapsed_time}"
            )

        self.genotype_encoder_training_losses = training_losses

    def plot_reconstruction_losses(self) -> None:
        """
        Plot all of the training losses.
        """
        plt.plot(self.phenotype_encoder_decoder_training_losses, label="Phenotype encoder-decoder")
        plt.plot(self.genotype_encoder_training_losses, label="Genotype encoder")
        plt.title("Training losses")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(self.config.dataset_path + "reconstruction_loss.svg")
        plt.close()


def analyze_predictions(
    phens,
    phen_encodings,
    phen_latent,
    fa_attr,
    dataset_path,
    n_phens_pred,
    model_type="g_p",
):
    """
    Evaluate the performance of each model, saving summaries of model performance.

    Parameters
    ----------
    phens : list
        List of real phenotype values (numpy arrays) from model evaluation
    phen_encodings : list
        List of predicted phenotype values (numpy arrays) from model evaluation
    phen_latent : list
        List of latent space representations (numpy arrays) from model evaluation
    fa_attr : list
        List of feature attribution values (numpy arrays) from captum analysis
    dataset_path : str
        Path to directory where all output files should be saved
    n_phens_pred : int
        Number of phenotypes to analyze and plot (uses first n_phens_pred phenotypes)
    model_type : str, optional
        Type of prediction being analyzed - either 'g_p' (genotype-phenotype) or
        'p_p' (phenotype-phenotype). Affects file naming and data saving. Default is 'g_p'

    Returns
    -------
    list
        List of metric results, containing in order: Pearson correlations, MSE values,
        MAPE values, and R2 scores for the first n_phens_pred phenotypes
    """
    suffix = "_p" if model_type == "p_p" else ""

    # Save attributions
    plt.hist(fa_attr, bins=20)
    plt.savefig(dataset_path + f"{model_type}_attr.svg")
    plt.close()
    if fa_attr != []:
        pickle.dump(fa_attr, open(dataset_path + f"{model_type}_attr.pk", "wb"))

    # Convert and transpose data
    phens = np.array(phens).T
    phen_encodings = np.array(phen_encodings).T

    # Save predictions data
    if model_type == "p_p":
        phen_latent = np.array(phen_latent).T
        pickle.dump(
            [phens, phen_encodings, phen_latent],
            open(dataset_path + f"phens_phen_encodings_dng_attr{suffix}.pk", "wb"),
        )
    else:
        pickle.dump(
            [phens, phen_encodings],
            open(dataset_path + f"phens_phen_encodings_dng_attr{suffix}.pk", "wb"),
        )

    # Plot predictions
    for n in range(len(phens[:n_phens_pred])):
        plt.plot(phens[n], phen_encodings[n], "o")
    plt.xlabel("real")
    plt.ylabel("predicted")
    plt.gca().set_aspect("equal")
    plt.savefig(dataset_path + f"phen_real_pred_dng_attr{suffix}.svg")
    plt.close()

    # Calculate and plot metrics
    stats_aggregator = []

    # Pearson correlation
    cors = [
        sc.stats.pearsonr(phens[n], phen_encodings[n])[0] for n in range(len(phens[:n_phens_pred]))
    ]
    stats_aggregator.append(cors)
    plt.hist(cors, bins=20)
    plt.savefig(dataset_path + f"phen_real_pred_pearsonsr_dng_attr{suffix}.svg")
    plt.close()

    # MSE
    errs = [
        mean_squared_error(phens[n], phen_encodings[n]) for n in range(len(phens[:n_phens_pred]))
    ]
    stats_aggregator.append(errs)
    plt.hist(errs, bins=20)
    plt.savefig(dataset_path + f"phen_real_pred_mse_dng_attr{suffix}.svg")
    plt.close()

    # MAPE
    errs = [
        mean_absolute_percentage_error(phens[n], phen_encodings[n])
        for n in range(len(phens[:n_phens_pred]))
    ]
    stats_aggregator.append(errs)
    plt.hist(errs, bins=20)
    plt.savefig(dataset_path + f"phen_real_pred_mape_dng_attr{suffix}.svg")
    plt.close()

    # R2
    errs = [r2_score(phens[n], phen_encodings[n]) for n in range(len(phens[:n_phens_pred]))]
    stats_aggregator.append(errs)
    plt.hist(errs, bins=20)
    plt.savefig(dataset_path + f"phen_real_pred_r2_dng_attr{suffix}.svg")
    plt.close()

    return stats_aggregator


def evaluate_model():
    """
    TODO: figure out where this should live.
    """
    out_stats = open(dataset_path + "test_stats.pk", "wb")
    stats_aggregator = []

    # G-P prediction. Loop evaluating the performance on the test data
    genotype_encoder.eval()
    phens, phen_encodings, phen_latent, fa_attr = [], [], [], []

    for dat in test_loader_geno:
        ph, gt = dat
        gt = gt[:, : args.n_loci_measured * args.n_alleles]
        ph = ph.to(device)
        gt = gt.to(device)
        batch_size = ph.shape[0]
        z_sample = genotype_encoder(gt)
        X_sample = phenotype_decoder(z_sample)
        phens += list(ph.detach().cpu().numpy())
        phen_encodings += list(X_sample.detach().cpu().numpy())
        phen_latent += list(z_sample.detach().cpu().numpy())
        if args.calculate_importance == "yes":
            fa_attr.append(
                list(
                    genotype_feature_ablation.attribute(inputs=(gt, ph))[0]
                    .squeeze()
                    .detach()
                    .cpu()
                    .numpy()
                )
            )

    stats_aggregator.extend(
        analyze_predictions(
            phens, phen_encodings, phen_latent, fa_attr, dataset_path, n_phens_pred, "g_p"
        )
    )

    # P-P prediction. Loop evaluating the performance on the test data
    phenotype_encoder.eval()
    phens, phen_encodings, phen_latent, fa_attr = [], [], [], []

    for dat in test_loader_pheno:
        ph = dat
        ph = ph.to(device)
        batch_size = ph.shape[0]
        z_sample = phenotype_encoder(ph)
        X_sample = phenotype_decoder(z_sample)
        phens += list(ph.detach().cpu().numpy())
        phen_encodings += list(X_sample.detach().cpu().numpy())
        phen_latent += list(z_sample.detach().cpu().numpy())
        if args.calculate_importance == "yes":
            fa_attr.append(
                list(
                    phenotype_feature_ablation.attribute(inputs=(ph, ph))[0]
                    .squeeze()
                    .detach()
                    .cpu()
                    .numpy()
                )
            )

    stats_aggregator.extend(
        analyze_predictions(
            phens, phen_encodings, phen_latent, fa_attr, dataset_path, n_phens_pred, "p_p"
        )
    )

    # Save and close stats
    pickle.dump(stats_aggregator, out_stats)
    out_stats.close()


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

    val_dataset = GPAtlasDataset(
        filepath=(dataset_filepath + "test.pk"), num_phenotypes=model_config.num_input_phenotypes
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=trainer_config.batch_size,
        num_workers=trainer_config.num_cores,
        shuffle=True,
    )

    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=1,
        num_workers=trainer_config.num_cores,
        shuffle=True,
    )

    model = GPAtlasModel(config=model_config)

    if trainer_config.hot_start:
        model.load_from_checkpoint(dirpath=trainer_config.hot_start_dirpath)

    trainer = GPAtlasTrainer(
        model=model, config=trainer_config, train_loader=train_loader, device=device
    )

    trainer.train_phenotype_encoder_decoder()

    # Freeze the phenotype encoder and decoder before training the genotype encoder.
    model.freeze_phenotype_encoder_decoder()

    trainer.train_genotype_encoder()

    plot_reconstruction_losses()

    evaluate_model()


if __name__ == "__main__":
    main()
