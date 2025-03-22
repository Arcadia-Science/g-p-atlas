#!/usr/bin/env python3
"""
G-P Atlas: Genotype-Phenotype Mapping

This module implements the G-P Atlas method for mapping genotype to phenotype
as described in https://doi.org/10.57844/arcadia-d316-721f.

The script trains a neural network architecture to:
1. Create a latent encoding of phenotypes using an autoencoder
2. Map genotypes to this same latent space for prediction

For help, run:
    python3 g_p_atlas_refactored.py --help
"""

import pickle
import sys
import time
from argparse import ArgumentParser
from typing import Dict, List, Optional, Tuple, Union, Any

import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
from captum.attr import FeatureAblation
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import DataLoader, Dataset


# Constants
EPS = 1e-15  # Minimum value to avoid divide by zero and buffer underflow


def parse_arguments() -> ArgumentParser:
    """Parse command-line arguments for the G-P Atlas model.
    
    Returns:
        ArgumentParser: Parsed command-line arguments
    """
    parser = ArgumentParser(description="G-P Atlas: Genotype-Phenotype Mapping")
    
    # Genetic configuration
    parser.add_argument(
        "--n_alleles", 
        type=int, 
        default=2,
        help="Number of segregating causal alleles at any given causal locus"
    )
    parser.add_argument(
        "--n_locs", 
        type=int, 
        default=900,
        help="Number of causal loci to model. This is the same as the number of genes"
    )
    parser.add_argument(
        "--n_env", 
        type=int, 
        default=3, 
        help="Number of influential continuous components"
    )
    
    # Phenotype configuration
    parser.add_argument(
        "--n_phens", 
        type=int, 
        default=30, 
        help="Number of phenotypes"
    )
    parser.add_argument(
        "--n_phens_to_analyze", 
        type=int, 
        default=30, 
        help="Number of phenotypes to analyze"
    )
    parser.add_argument(
        "--n_phens_to_predict", 
        type=int, 
        default=30, 
        help="Number of phenotypes to predict"
    )
    
    # Loss weights
    parser.add_argument(
        "--gen_lw", 
        type=float, 
        default=1, 
        help="Weight for the loss attributed to genetic features"
    )
    parser.add_argument(
        "--eng_lw", 
        type=float, 
        default=0.1, 
        help="Weight for the loss attributed to env features"
    )

    # Training configuration
    parser.add_argument(
        "--n_epochs", 
        type=int, 
        default=100, 
        help="Number of epochs for training the phenotype autoencoder"
    )
    parser.add_argument(
        "--n_epochs_gen", 
        type=int, 
        default=100, 
        help="Number of epochs to train the genetic encoder"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=16, 
        help="Batch size"
    )
    parser.add_argument(
        "--lr_r", 
        type=float, 
        default=0.001, 
        help="Reconstruction learning rate"
    )
    parser.add_argument(
        "--b1", 
        type=float, 
        default=0.5, 
        help="Adam optimizer beta1 parameter"
    )
    parser.add_argument(
        "--b2", 
        type=float, 
        default=0.999, 
        help="Adam optimizer beta2 parameter"
    )
    parser.add_argument(
        "--n_cpu", 
        type=int, 
        default=14, 
        help="Number of CPU workers for data loading"
    )
    
    # Model architecture
    parser.add_argument(
        "--e_hidden_dim", 
        type=int, 
        default=32, 
        help="Number of neurons in the hidden layers of encoder"
    )
    parser.add_argument(
        "--d_hidden_dim", 
        type=int, 
        default=32, 
        help="Number of neurons in the hidden layers of decoder"
    )
    parser.add_argument(
        "--ge_hidden_dim", 
        type=int, 
        default=32,
        help="Number of neurons in the hidden layers of the genetic encoder"
    )
    parser.add_argument(
        "--latent_dim", 
        type=int, 
        default=32, 
        help="Number of neurons in the latent space"
    )
    parser.add_argument(
        "--batchnorm_momentum", 
        type=float, 
        default=0.8,
        help="Momentum for the batch normalization layers"
    )
    
    # Data noise parameters
    parser.add_argument(
        "--sd_noise", 
        type=float, 
        default=0.1, 
        help="Noise added to phenotypes during training"
    )
    parser.add_argument(
        "--gen_noise", 
        type=float, 
        default=0.3, 
        help="Noise added to genotypes during training"
    )
    
    # Data configuration
    parser.add_argument(
        "--n_loci_measured", 
        type=int, 
        default=3000,
        help="Number of genetic loci for which there is data"
    )
    
    # Regularization
    parser.add_argument(
        "--l1_lambda", 
        type=float, 
        default=0.8, 
        help="L1 regularization weight for the genetic weights"
    )
    parser.add_argument(
        "--l2_lambda", 
        type=float, 
        default=0.01,
        help="L2 regularization weight for the genetic weights"
    )
    
    # I/O configuration
    parser.add_argument(
        "--dataset_path", 
        type=str, 
        default=None, 
        help="Directory where the train and test files are located"
    )
    parser.add_argument(
        "--train_suffix", 
        type=str, 
        default="train.pk",
        help="Name of the training data file, defaults to DGRP dataset"
    )
    parser.add_argument(
        "--test_suffix", 
        type=str, 
        default="test.pk", 
        help="Name of the test data file"
    )
    
    # Model initialization
    parser.add_argument(
        "--hot_start", 
        type=bool, 
        default=False,
        help="Flag to use precomputed weights, false by default"
    )
    parser.add_argument(
        "--hot_start_path_e", 
        type=str, 
        default=None,
        help="Path to the phenotype encoder weights to use at initialization"
    )
    parser.add_argument(
        "--hot_start_path_d", 
        type=str, 
        default=None,
        help="Path to the phenotype decoder weights to use at initialization"
    )
    parser.add_argument(
        "--hot_start_path_ge", 
        type=str, 
        default=None,
        help="Path to the genotype encoder weights to use at initialization"
    )
    
    # Feature importance
    parser.add_argument(
        "--calculate_importance", 
        type=str, 
        default="no",
        help="Flag whether to calculate variable importance. Expects 'no' or 'yes'"
    )
    
    return parser


class PhenotypeDataset(Dataset):
    """Dataset for loading and accessing phenotype data.
    
    This class handles loading phenotype data from pickled files. The data
    is expected to be organized as a dictionary with a 'phenotypes' key
    containing a tensor of shape [n_animals, n_phenotypes].
    
    Attributes:
        dataset: The loaded pickled dataset
        phens: Tensor containing phenotype data
        data_file: Path to the data file
        n_phens: Number of phenotypes to use
    """
    
    def __init__(self, data_file: str, n_phens: int):
        """Initialize the phenotype dataset.
        
        Args:
            data_file: Path to the pickled data file
            n_phens: Number of phenotypes to use
        """
        with open(data_file, "rb") as f:
            self.dataset = pickle.load(f)
        
        self.phens = torch.tensor(np.array(self.dataset["phenotypes"]), dtype=torch.float32)
        self.data_file = data_file
        self.n_phens = n_phens

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.phens)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            The phenotype tensor for the sample
        """
        return self.phens[idx][:self.n_phens]


class GenotypeDataset(Dataset):
    """Dataset for loading and accessing genotype and phenotype data.
    
    This class handles loading genotype and phenotype data from pickled files.
    The data is expected to be organized as a dictionary with:
    - 'genotypes': tensor of shape [n_animals, n_loci, n_alleles]
    - 'phenotypes': tensor of shape [n_animals, n_phenotypes]
    
    Attributes:
        dataset: The loaded pickled dataset
        phens: Tensor containing phenotype data
        genotypes: Tensor containing genotype data
        data_file: Path to the data file
        n_geno: Number of genotype loci to use
        n_phens: Number of phenotypes to use
    """
    
    def __init__(self, data_file: str, n_geno: int, n_phens: int):
        """Initialize the genotype+phenotype dataset.
        
        Args:
            data_file: Path to the pickled data file
            n_geno: Number of genotype loci to use
            n_phens: Number of phenotypes to use
        """
        with open(data_file, "rb") as f:
            self.dataset = pickle.load(f)
            
        self.phens = torch.tensor(np.array(self.dataset["phenotypes"]), dtype=torch.float32)
        self.genotypes = torch.tensor(np.array(self.dataset["genotypes"]), dtype=torch.float32)
        self.data_file = data_file
        self.n_geno = n_geno
        self.n_phens = n_phens

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.genotypes)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple containing (phenotype_tensor, flattened_genotype_tensor)
        """
        phenotypes = self.phens[idx][:self.n_phens]
        genotype = torch.flatten(self.genotypes[idx])
        return phenotypes, genotype


class PhenotypeEncoder(nn.Module):
    """Neural network for encoding phenotypic data into latent space.
    
    This network transforms phenotype data into a lower-dimensional latent
    representation that captures essential phenotypic patterns.
    
    Attributes:
        encoder: Sequential neural network for encoding
    """
    
    def __init__(self, phen_dim: Optional[int] = None, hidden_dim: Optional[int] = None, 
                 latent_dim: int = 32, batchnorm_momentum: float = 0.8):
        """Initialize the phenotype encoder.
        
        Args:
            phen_dim: Input dimension (number of phenotypes)
            hidden_dim: Number of neurons in hidden layer
            latent_dim: Dimension of the latent space
            batchnorm_momentum: Momentum for batch normalization
        """
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(in_features=phen_dim, out_features=hidden_dim),
            nn.BatchNorm1d(hidden_dim, momentum=batchnorm_momentum),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(in_features=hidden_dim, out_features=latent_dim),
            nn.BatchNorm1d(latent_dim, momentum=batchnorm_momentum),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the encoder.
        
        Args:
            x: Input phenotype tensor
            
        Returns:
            Encoded latent representation
        """
        return self.encoder(x)


class PhenotypeDecoder(nn.Module):
    """Neural network for decoding latent space back to phenotypes.
    
    This network transforms the latent representation back into
    phenotype space, allowing prediction of phenotypes from the encoding.
    
    Attributes:
        decoder: Sequential neural network for decoding
    """
    
    def __init__(self, out_phen_dim: int, hidden_dim: int, 
                 latent_dim: int = 32, batchnorm_momentum: float = 0.8):
        """Initialize the phenotype decoder.
        
        Args:
            out_phen_dim: Output dimension (number of phenotypes to predict)
            hidden_dim: Number of neurons in hidden layer
            latent_dim: Dimension of the latent space
            batchnorm_momentum: Momentum for batch normalization
        """
        super().__init__()
        
        self.decoder = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=hidden_dim),
            nn.BatchNorm1d(hidden_dim, momentum=batchnorm_momentum),
            nn.LeakyReLU(0.01),
            nn.Linear(in_features=hidden_dim, out_features=out_phen_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the decoder.
        
        Args:
            x: Input latent representation tensor
            
        Returns:
            Decoded phenotype predictions
        """
        return self.decoder(x)


class GeneticEncoder(nn.Module):
    """Neural network for encoding genetic data into the shared latent space.
    
    This network transforms genotype data into the same latent space used
    by the phenotype encoder/decoder, enabling prediction of phenotypes
    directly from genotypes.
    
    Attributes:
        encoder: Sequential neural network for encoding
    """
    
    def __init__(self, n_loci: int, hidden_dim: int, 
                 latent_dim: int = 32, batchnorm_momentum: float = 0.8):
        """Initialize the genetic encoder.
        
        Args:
            n_loci: Input dimension (number of genetic loci * number of alleles)
            hidden_dim: Number of neurons in hidden layer
            latent_dim: Dimension of the latent space
            batchnorm_momentum: Momentum for batch normalization
        """
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(in_features=n_loci, out_features=hidden_dim),
            nn.BatchNorm1d(hidden_dim, momentum=batchnorm_momentum),
            nn.LeakyReLU(0.01),
            nn.Linear(in_features=hidden_dim, out_features=latent_dim),
            nn.BatchNorm1d(latent_dim, momentum=batchnorm_momentum),
            nn.LeakyReLU(0.01),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the encoder.
        
        Args:
            x: Input genotype tensor
            
        Returns:
            Encoded latent representation
        """
        return self.encoder(x)


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Percentage Error between true and predicted values.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        MAPE value as a percentage
    """
    y_true, y_pred = np.array(y_true + EPS), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def analyze_predictions(
    phens: List[np.ndarray],
    phen_encodings: List[np.ndarray],
    phen_latent: List[np.ndarray],
    fa_attr: List[np.ndarray],
    dataset_path: str,
    n_phens_pred: int,
    model_type: str = "g_p"
) -> List[List[float]]:
    """Analyze predictions and save visualization results.

    Args:
        phens: List of real phenotype values from model evaluation
        phen_encodings: List of predicted phenotype values from model evaluation
        phen_latent: List of latent space representations from model evaluation
        fa_attr: List of feature attribution values from captum analysis
        dataset_path: Path to directory where output files should be saved
        n_phens_pred: Number of phenotypes to analyze and plot
        model_type: Type of prediction ('g_p' for genotype-phenotype or 
                    'p_p' for phenotype-phenotype)

    Returns:
        List of metric results: [Pearson correlations, MSE values, MAPE values, R2 scores]
    """
    suffix = "_p" if model_type == "p_p" else ""

    # Save attributions
    if fa_attr:
        plt.hist(fa_attr, bins=20)
        plt.savefig(dataset_path + f"{model_type}_attr.svg")
        plt.close()
        with open(dataset_path + f"{model_type}_attr.pk", "wb") as f:
            pickle.dump(fa_attr, f)

    # Convert and transpose data
    phens = np.array(phens).T
    phen_encodings = np.array(phen_encodings).T

    # Save predictions data
    if model_type == "p_p":
        phen_latent = np.array(phen_latent).T
        with open(dataset_path + f"phens_phen_encodings_dng_attr{suffix}.pk", "wb") as f:
            pickle.dump([phens, phen_encodings, phen_latent], f)
    else:
        with open(dataset_path + f"phens_phen_encodings_dng_attr{suffix}.pk", "wb") as f:
            pickle.dump([phens, phen_encodings], f)

    # Plot predictions
    for n in range(min(len(phens), n_phens_pred)):
        plt.plot(phens[n], phen_encodings[n], "o")
    plt.xlabel("real")
    plt.ylabel("predicted")
    plt.gca().set_aspect("equal")
    plt.savefig(dataset_path + f"phen_real_pred_dng_attr{suffix}.svg")
    plt.close()

    # Calculate and plot metrics
    stats_aggregator = []
    num_phens_to_analyze = min(len(phens), n_phens_pred)

    # Pearson correlation
    cors = [
        sc.stats.pearsonr(phens[n], phen_encodings[n])[0] 
        for n in range(num_phens_to_analyze)
    ]
    stats_aggregator.append(cors)
    plt.hist(cors, bins=20)
    plt.savefig(dataset_path + f"phen_real_pred_pearsonsr_dng_attr{suffix}.svg")
    plt.close()

    # MSE
    errs = [
        mean_squared_error(phens[n], phen_encodings[n]) 
        for n in range(num_phens_to_analyze)
    ]
    stats_aggregator.append(errs)
    plt.hist(errs, bins=20)
    plt.savefig(dataset_path + f"phen_real_pred_mse_dng_attr{suffix}.svg")
    plt.close()

    # MAPE
    errs = [
        mean_absolute_percentage_error(phens[n], phen_encodings[n])
        for n in range(num_phens_to_analyze)
    ]
    stats_aggregator.append(errs)
    plt.hist(errs, bins=20)
    plt.savefig(dataset_path + f"phen_real_pred_mape_dng_attr{suffix}.svg")
    plt.close()

    # R2
    errs = [
        r2_score(phens[n], phen_encodings[n]) 
        for n in range(num_phens_to_analyze)
    ]
    stats_aggregator.append(errs)
    plt.hist(errs, bins=20)
    plt.savefig(dataset_path + f"phen_real_pred_r2_dng_attr{suffix}.svg")
    plt.close()

    return stats_aggregator


class GPAtlasModel:
    """Main class for the G-P Atlas model.
    
    This class handles the training and evaluation of the G-P Atlas model,
    which maps genotypes to phenotypes through a shared latent space.
    
    Attributes:
        args: Command-line arguments
        device: Device for model execution (CPU or CUDA)
        phenotype_encoder: Neural network for encoding phenotypes
        phenotype_decoder: Neural network for decoding phenotypes from latent space
        genetic_encoder: Neural network for encoding genotypes to the same latent space
    """
    
    def __init__(self, args: ArgumentParser):
        """Initialize the G-P Atlas model.
        
        Args:
            args: Parsed command-line arguments
        """
        self.args = args
        
        # Set up device (GPU if available, otherwise CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize models
        self.phenotype_encoder = PhenotypeEncoder(
            phen_dim=args.n_phens_to_analyze,
            hidden_dim=args.e_hidden_dim,
            latent_dim=args.latent_dim,
            batchnorm_momentum=args.batchnorm_momentum
        )
        
        self.phenotype_decoder = PhenotypeDecoder(
            out_phen_dim=args.n_phens_to_predict,
            hidden_dim=args.d_hidden_dim,
            latent_dim=args.latent_dim,
            batchnorm_momentum=args.batchnorm_momentum
        )
        
        self.genetic_encoder = GeneticEncoder(
            n_loci=args.n_loci_measured * args.n_alleles,
            hidden_dim=args.ge_hidden_dim,
            latent_dim=args.latent_dim,
            batchnorm_momentum=args.batchnorm_momentum
        )
        
        # Load pre-computed weights if requested
        if args.hot_start:
            if args.hot_start_path_e:
                self.phenotype_encoder.load_state_dict(torch.load(args.hot_start_path_e), strict=False)
            if args.hot_start_path_d:
                self.phenotype_decoder.load_state_dict(torch.load(args.hot_start_path_d), strict=False)
            if args.hot_start_path_ge:
                self.genetic_encoder.load_state_dict(torch.load(args.hot_start_path_ge), strict=False)
        
        # Move models to device
        self.phenotype_encoder.to(self.device)
        self.phenotype_decoder.to(self.device)
        self.genetic_encoder.to(self.device)
        
        # Set up optimizers
        adam_betas = (args.b1, args.b2)
        self.optimizer_decoder = torch.optim.Adam(
            self.phenotype_decoder.parameters(), 
            lr=args.lr_r, 
            betas=adam_betas
        )
        self.optimizer_phenotype_encoder = torch.optim.Adam(
            self.phenotype_encoder.parameters(), 
            lr=args.lr_r, 
            betas=adam_betas
        )
        self.optimizer_genetic_encoder = torch.optim.Adam(
            self.genetic_encoder.parameters(), 
            lr=args.lr_r, 
            betas=adam_betas
        )
        
        # Set up feature attribution methods
        self.fa_genotype = FeatureAblation(self._sequential_forward_attr_gen_phen)
        self.fa_phenotype = FeatureAblation(self._sequential_forward_attr_phen_phen)
        
        # Set seed for reproducibility
        torch.manual_seed(47)

    def _sequential_forward_attr_gen_phen(self, input: torch.Tensor, phens: torch.Tensor) -> torch.Tensor:
        """Forward function for genotype-phenotype feature attribution.
        
        This function is used by the captum library to calculate feature importance.
        
        Args:
            input: Input genotype tensor
            phens: Target phenotype tensor
            
        Returns:
            Loss value between predicted and target phenotypes
        """
        latent = self.genetic_encoder(input)
        predictions = self.phenotype_decoder(latent)
        return F.mse_loss(predictions + EPS, phens[:, :self.args.n_phens_to_predict] + EPS)

    def _sequential_forward_attr_phen_phen(self, input: torch.Tensor, phens: torch.Tensor) -> torch.Tensor:
        """Forward function for phenotype-phenotype feature attribution.
        
        This function is used by the captum library to calculate feature importance.
        
        Args:
            input: Input phenotype tensor
            phens: Target phenotype tensor
            
        Returns:
            Loss value between predicted and target phenotypes
        """
        latent = self.phenotype_encoder(input)
        predictions = self.phenotype_decoder(latent)
        return F.mse_loss(predictions + EPS, phens[:, :self.args.n_phens_to_predict] + EPS)

    def train_phenotype_autoencoder(self, train_loader: DataLoader) -> List[float]:
        """Train the phenotype encoder and decoder.
        
        Args:
            train_loader: DataLoader containing phenotype data
            
        Returns:
            List of reconstruction loss values for each batch
        """
        reconstruction_losses = []
        n_phens = self.args.n_phens_to_analyze
        n_phens_pred = self.args.n_phens_to_predict
        
        for epoch in range(self.args.n_epochs):
            start_time = time.time()
            
            for batch_idx, phens in enumerate(train_loader):
                # Use only the specified number of phenotypes
                phens = phens[:, :n_phens].to(self.device)
                batch_size = phens.shape[0]
                
                # Zero gradients
                self.phenotype_decoder.zero_grad()
                self.phenotype_encoder.zero_grad()
                
                # Add noise to phenotypes for robustness
                noise_phens = phens + (self.args.sd_noise**0.5) * torch.randn(phens.shape).to(self.device)
                
                # Forward pass
                latent = self.phenotype_encoder(noise_phens)
                predictions = self.phenotype_decoder(latent)
                
                # Calculate loss and backpropagate
                recon_loss = F.mse_loss(predictions + EPS, phens[:, :n_phens_pred] + EPS)
                reconstruction_losses.append(float(recon_loss.detach()))
                
                recon_loss.backward()
                self.optimizer_decoder.step()
                self.optimizer_phenotype_encoder.step()
            
            # Calculate and report time per epoch
            epoch_time = time.time() - start_time
            print(f"Epoch: {epoch}, Batch: {batch_idx}, "
                  f"Reconstruction loss: {reconstruction_losses[-1]:.6f}, "
                  f"Time: {epoch_time:.2f}s")
        
        return reconstruction_losses

    def train_genetic_encoder(self, train_loader: DataLoader) -> List[float]:
        """Train the genetic encoder model.
        
        Args:
            train_loader: DataLoader containing genotype and phenotype data
            
        Returns:
            List of reconstruction loss values for each batch
        """
        # Freeze the phenotype decoder - we're only training the genetic encoder
        self.phenotype_decoder.requires_grad_(False)
        self.phenotype_decoder.eval()
        
        reconstruction_losses = []
        n_phens_pred = self.args.n_phens_to_predict
        gen_noise = 1 - self.args.gen_noise
        
        for epoch in range(self.args.n_epochs_gen):
            start_time = time.time()
            
            for batch_idx, (phens, gens) in enumerate(train_loader):
                phens = phens.to(self.device)
                
                # Limit genotypes to measured loci
                gens = gens[:, :self.args.n_loci_measured * self.args.n_alleles]
                
                # Add noise to genotypes for robustness
                pos_noise = np.random.binomial(1, gen_noise / 2, gens.shape)
                neg_noise = np.random.binomial(1, gen_noise / 2, gens.shape)
                noise_gens = torch.tensor(
                    np.where((gens + pos_noise - neg_noise) > 0, 1, 0), 
                    dtype=torch.float32
                ).to(self.device)
                
                batch_size = phens.shape[0]
                
                # Zero gradients
                self.genetic_encoder.zero_grad()
                
                # Forward pass
                latent = self.genetic_encoder(noise_gens)
                predictions = self.phenotype_decoder(latent)
                
                # Calculate loss
                g_recon_loss = F.mse_loss(predictions + EPS, phens[:, :n_phens_pred] + EPS)
                
                # Add L1 and L2 regularization to the weights
                l1_reg = torch.linalg.norm(torch.sum(self.genetic_encoder.encoder[0].weight, axis=0), 1)
                l2_reg = torch.linalg.norm(torch.sum(self.genetic_encoder.encoder[0].weight, axis=0), 2)
                g_recon_loss = g_recon_loss + l1_reg * self.args.l1_lambda + l2_reg * self.args.l2_lambda
                
                reconstruction_losses.append(float(g_recon_loss.detach()))
                
                # Backpropagate and optimize
                g_recon_loss.backward()
                self.optimizer_genetic_encoder.step()
            
            # Calculate and report time per epoch
            epoch_time = time.time() - start_time
            print(f"Epoch: {epoch}, Batch: {batch_idx}, "
                  f"Reconstruction loss: {reconstruction_losses[-1]:.6f}, "
                  f"Time: {epoch_time:.2f}s")
        
        return reconstruction_losses

    def evaluate_genotype_predictions(self, test_loader: DataLoader) -> List[List[float]]:
        """Evaluate genotype to phenotype predictions on test data.
        
        Args:
            test_loader: DataLoader with test genotype and phenotype data
            
        Returns:
            List of metric results: [Pearson correlations, MSE values, MAPE values, R2 scores]
        """
        self.genetic_encoder.eval()
        phens, phen_encodings, phen_latent, fa_attr = [], [], [], []
        
        with torch.no_grad():
            for ph, gt in test_loader:
                gt = gt[:, :self.args.n_loci_measured * self.args.n_alleles].to(self.device)
                ph = ph.to(self.device)
                
                latent = self.genetic_encoder(gt)
                predictions = self.phenotype_decoder(latent)
                
                phens.append(ph.cpu().numpy())
                phen_encodings.append(predictions.cpu().numpy())
                phen_latent.append(latent.cpu().numpy())
                
                # Calculate feature importance if requested
                if self.args.calculate_importance == "yes":
                    fa_attr.append(
                        self.fa_genotype.attribute(inputs=(gt, ph))[0].squeeze().cpu().numpy()
                    )
        
        return analyze_predictions(
            phens, 
            phen_encodings, 
            phen_latent, 
            fa_attr, 
            self.args.dataset_path, 
            self.args.n_phens_to_predict, 
            "g_p"
        )

    def evaluate_phenotype_predictions(self, test_loader: DataLoader) -> List[List[float]]:
        """Evaluate phenotype to phenotype predictions on test data.
        
        Args:
            test_loader: DataLoader with test phenotype data
            
        Returns:
            List of metric results: [Pearson correlations, MSE values, MAPE values, R2 scores]
        """
        self.phenotype_encoder.eval()
        phens, phen_encodings, phen_latent, fa_attr = [], [], [], []
        
        with torch.no_grad():
            for ph in test_loader:
                ph = ph.to(self.device)
                
                latent = self.phenotype_encoder(ph)
                predictions = self.phenotype_decoder(latent)
                
                phens.append(ph.cpu().numpy())
                phen_encodings.append(predictions.cpu().numpy())
                phen_latent.append(latent.cpu().numpy())
                
                # Calculate feature importance if requested
                if self.args.calculate_importance == "yes":
                    fa_attr.append(
                        self.fa_phenotype.attribute(inputs=(ph, ph))[0].squeeze().cpu().numpy()
                    )
        
        return analyze_predictions(
            phens, 
            phen_encodings, 
            phen_latent, 
            fa_attr, 
            self.args.dataset_path, 
            self.args.n_phens_to_predict, 
            "p_p"
        )

    def save_models(self) -> None:
        """Save model states to disk."""
        torch.save(self.phenotype_encoder.state_dict(), 
                  self.args.dataset_path + "phen_encoder_state.pt")
        torch.save(self.phenotype_decoder.state_dict(), 
                  self.args.dataset_path + "phen_decoder_state.pt")
        torch.save(self.genetic_encoder.state_dict(), 
                  self.args.dataset_path + "gen_encoder_state.pt")

    def plot_reconstruction_loss(self, pheno_losses: List[float], geno_losses: List[float]) -> None:
        """Plot and save the reconstruction losses.
        
        Args:
            pheno_losses: List of phenotype reconstruction losses
            geno_losses: List of genotype reconstruction losses
        """
        plt.figure()
        plt.plot(pheno_losses, label="Phenotype autoencoder")
        plt.plot(geno_losses, label="Genetic encoder")
        plt.legend()
        plt.savefig(self.args.dataset_path + "reconstruction_loss.svg")
        plt.close()


def main() -> None:
    """Main function to run the G-P Atlas model."""
    # Parse arguments
    parser = parse_arguments()
    args = parser.parse_args()
    
    # Save command line parameters
    if args.dataset_path:
        with open(args.dataset_path + "run_params.txt", "w") as params_file:
            params_file.write(" ".join(sys.argv[:]))
    
    # Load datasets
    train_data_pheno = PhenotypeDataset(
        args.dataset_path + args.train_suffix, 
        n_phens=args.n_phens_to_analyze
    )
    test_data_pheno = PhenotypeDataset(
        args.dataset_path + args.test_suffix, 
        n_phens=args.n_phens_to_analyze
    )
    
    train_data_geno = GenotypeDataset(
        args.dataset_path + args.train_suffix,
        n_geno=args.n_loci_measured,
        n_phens=args.n_phens_to_analyze
    )
    test_data_geno = GenotypeDataset(
        args.dataset_path + args.test_suffix,
        n_geno=args.n_loci_measured,
        n_phens=args.n_phens_to_analyze
    )
    
    # Create data loaders
    train_loader_pheno = DataLoader(
        dataset=train_data_pheno,
        batch_size=args.batch_size,
        num_workers=args.n_cpu,
        shuffle=True
    )
    test_loader_pheno = DataLoader(
        dataset=test_data_pheno,
        batch_size=1,
        num_workers=args.n_cpu,
        shuffle=True
    )
    
    train_loader_geno = DataLoader(
        dataset=train_data_geno,
        batch_size=args.batch_size,
        num_workers=args.n_cpu,
        shuffle=True
    )
    test_loader_geno = DataLoader(
        dataset=test_data_geno,
        batch_size=1,
        num_workers=args.n_cpu,
        shuffle=True
    )
    
    # Initialize and train the model
    model = GPAtlasModel(args)
    
    # Train the phenotype autoencoder
    print("Training phenotype autoencoder...")
    pheno_losses = model.train_phenotype_autoencoder(train_loader_pheno)
    
    # Train the genetic encoder
    print("Training genetic encoder...")
    geno_losses = model.train_genetic_encoder(train_loader_geno)
    
    # Plot reconstruction losses
    model.plot_reconstruction_loss(pheno_losses, geno_losses)
    
    # Save models
    model.save_models()
    
    # Evaluate models
    print("Evaluating genotype to phenotype predictions...")
    gp_stats = model.evaluate_genotype_predictions(test_loader_geno)
    
    print("Evaluating phenotype to phenotype predictions...")
    pp_stats = model.evaluate_phenotype_predictions(test_loader_pheno)
    
    # Save stats
    stats_aggregator = []
    stats_aggregator.extend(gp_stats)
    stats_aggregator.extend(pp_stats)
    
    with open(args.dataset_path + "test_stats.pk", "wb") as out_stats:
        pickle.dump(stats_aggregator, out_stats)
    
    print("Analysis complete. Results saved to", args.dataset_path)


if __name__ == "__main__":
    main()
