from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn as nn

# argument parser and default variables
args = ArgumentParser()
args.add_argument(
    "--n_alleles",
    type=int,
    default=3,
    help="number of segregating causal alleles at any given causal locus",
)
args.add_argument(
    "--n_locs",
    type=int,
    default=900,
    help="number of causal loci to model.  This is the same as the number of genes",
)
args.add_argument(
    "--n_env", type=int, default=3, help="number of influential continuous components"
)
args.add_argument("--n_phens", type=int, default=30, help="number of phenotypes")
args.add_argument(
    "--gen_lw",
    type=float,
    default=1,
    help="weight for the loss attributed to genetic features",
)
args.add_argument(
    "--eng_lw",
    type=float,
    default=0.1,
    help="weight for the loss attributed to env features",
)
args.add_argument(
    "--n_epochs", type=int, default=100, help="number of epochs of training"
)
args.add_argument("--batch_size", type=int, default=16, help="batch size")
args.add_argument(
    "--lr_r", type=float, default=0.001, help="reconstruction learning rate"
)
args.add_argument(
    "--b1", type=float, default=0.5, help="adam: gradient decay variables"
)
args.add_argument(
    "--b2", type=float, default=0.999, help="adam: gradient decay variables"
)
args.add_argument("--n_cpu", type=int, default=14, help="number of cpus")
args.add_argument(
    "--e_hidden_dim",
    type=int,
    default=64,
    help="number of neurons in the hidden layers of encoder",
)
args.add_argument(
    "--d_hidden_dim",
    type=int,
    default=64,
    help="number of neurons in the hidden layers of decoder",
)
args.add_argument(
    "--batchnorm_momentum",
    type=float,
    default=0.8,
    help="momentum for the batchnormalization layers",
)
args.add_argument(
    "--latent_dim", type=int, default=32, help="number of neurons in the latent space"
)
args.add_argument(
    "--n_phens_to_analyze", type=int, default=30, help="number of phenotypes to analyze"
)
args.add_argument("--sd_noise", type=float, default=0.1, help="noise added to phens")
args.add_argument("--gen_noise", type=float, default=0.1, help="noise added to gens")
args.add_argument(
    "--n_phens_to_predict", type=int, default=30, help="number of phenotypes to predict"
)
args.add_argument(
    "--n_epochs_gen",
    type=int,
    default=100,
    help="number of epochs to train the genetic encoder",
)
args.add_argument(
    "--n_loci_measured",
    type=int,
    default=3000,
    help="number of genetic loci for which there is data",
)
args.add_argument(
    "--l1_lambda",
    type=float,
    default=0.8,
    help="l1 regularlization weight for the genetic weights",
)
args.add_argument(
    "--l2_lambda",
    type=float,
    default=0.01,
    help="l2 regularlization weight for the genetic weights",
)
args.add_argument(
    "--device",
    type=str,
    default="cpu",
    help="Computational device to use. Can either be cpu or cuda",
)

variables = args.parse_args()

# setting device on GPU if available, else CPU
variables.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", variables.device)
print()


# encoders
"""A series of phenotype or genotype encoders"""


class PhenotypeEncoder(nn.Module):
    """Two layer phenotype encoder. Assumes all phenotypes are continuous
    random variables or are 1hot encoded"""

    def __init__(self, phen_dim=None, N=None):
        if N is None:
            N = variables.e_hidden_dim
        if phen_dim is None:
            phen_dim = variables.n_phens_to_analyze

        batchnorm_momentum = variables.batchnorm_momentum
        latent_dim = variables.latent_dim

        super().__init__() 
        self.encoder = nn.Sequential(
            nn.Linear(in_features=phen_dim, out_features=N),
            nn.BatchNorm1d(N, momentum=batchnorm_momentum),
            nn.LeakyReLU(0.01),
            nn.Linear(in_features=N, out_features=latent_dim),
            nn.LeakyReLU(0.01),
        )

    def forward(self, x):
        x = self.encoder(x)
        return x


class PhenotypeDenoisingEncoder(nn.Module):
    """Two layer denoising encoder. Adds gaussian noise to inputs training data"""

    def __init__(self, phen_dim=None, N=None):
        if N is None:
            N = variables.e_hidden_dim
        if phen_dim is None:
            phen_dim = variables.n_phens_to_analyze
        batchnorm_momentum = variables.batchnorm_momentum
        latent_dim = variables.latent_dim

        super().__init__() 
        self.encoder = nn.Sequential(
            nn.Linear(in_features=phen_dim, out_features=N),
            nn.BatchNorm1d(N, momentum=batchnorm_momentum),
            nn.LeakyReLU(0.01),
            nn.Linear(in_features=N, out_features=latent_dim),
            nn.LeakyReLU(0.01),
        )

    def forward(self, x):
        x += (variables.sd_noise**0.5) * torch.randn(
            x.shape
        )  # this might have to be sent to device
        x = self.encoder(x)
        return x


class PhenotypeEncoderCatGaussLatent(nn.Module):
    """An encoder with a latent space that has both categorical and normally
    distributed dimensions. It is intended to be used with an adversarial archtecture"""

    def __init__(self, phen_dim=None, N=None):
        super().__init__()
        if N is None:
            N = variables.e_hidden_dim
        if phen_dim is None:
            phen_dim = variables.n_phens_to_analyze

        self.batchnorm_momentum = variables.batchnorm_momentum
        self.n_allelic_states = variables.n_alleles
        self.n_environments = variables.n_env

        self.encoder = nn.Sequential(
            nn.Linear(in_features=phen_dim, out_features=N),
            nn.BatchNorm1d(N, momentum=self.batchnorm_momentum),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(in_features=N, out_features=N),
            nn.LeakyReLU(0.01, inplace=True),
        )

        # output layers
        self.cat_layer = nn.Sequential(nn.Linear(N, self.n_allelic_states))
        self.cat_layer2 = nn.Sequential(nn.Softmax(dim=2), nn.Flatten())
        self.norm_layer = nn.Sequential(nn.Linear(N, self.n_environments))

    def forward(self, x):
        x = self.encoder(x)
        cat_out = self.cat_layer(x)
        cat_out = torch.reshape(cat_out, (self.batch_size, self.n_locs, self.n_alleles))
        cat_out = self.cat_layer2(cat_out)
        norm_out = self.norm_layer(x)
        return (cat_out, norm_out)


class GenotypeEncoderTwoLayer(nn.Module):
    """Two layer genotype encoder. It expects 1 hot encoded genetic data"""

    def __init__(self, n_loci=None, N=None):
        super().__init__()
        if n_loci is None:
            n_loci = variables.n_loci_measured * variables.n_alleles

        batchnorm_momentum = variables.batchnorm_momentum
        latent_dim = variables.latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(in_features=n_loci, out_features=latent_dim),
            nn.BatchNorm1d(latent_dim, momentum=batchnorm_momentum),
            nn.LeakyReLU(0.01),
            nn.Linear(in_features=latent_dim, out_features=latent_dim),
            nn.LeakyReLU(0.01),
        )

    def forward(self, x):
        x = self.encoder(x)
        return x


class GenotypeDenoisingEncoder(nn.Module):
    """Two layer denoising genotype encoder. It adds genetically distributed noise
    to each training example"""

    def __init__(self, n_loci=None):
        super().__init__()
        if n_loci is None:
            n_loci = variables.n_loci_measured * variables.n_alleles
        self.gen_noise = variables.gen_noise
        batchnorm_momentum = variables.batchnorm_momentum
        latent_dim = variables.latent_dim
        self.device = variables.device
        self.encoder = nn.Sequential(
            nn.Linear(in_features=n_loci, out_features=latent_dim),
            nn.BatchNorm1d(latent_dim, momentum=batchnorm_momentum),
            nn.LeakyReLU(0.01),
        )

    def forward(self, x):
        pos_noise = np.random.binomial(1, self.gen_noise / 2, x.shape)
        neg_noise = np.random.binomial(1, self.gen_noise / 2, x.shape)
        noisey_genes = torch.tensor(
            np.where((x + pos_noise - neg_noise) > 0, 1, 0), dtype=torch.float32
        )
        x = noisey_genes.to(self.device)  # this might not need to be sent to device
        x = self.encoder(x)
        return x


# decoders
class PhenotypeDecoder(nn.Module):
    """Two layer phenotype decoder."""

    def __init__(self, phen_dim=None, N=None):
        super().__init__()
        if N is None:
            N = variables.d_hidden_dim
        if phen_dim is None:
            phen_dim = variables.n_phens_to_analyze
        out_phen_dim = variables.n_phens_to_predict
        latent_dim = variables.latent_dim
        batchnorm_momentum = variables.batchnorm_momentum
        self.decoder = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=N),
            nn.BatchNorm1d(N, momentum=batchnorm_momentum),
            nn.LeakyReLU(0.01),
            nn.Linear(in_features=N, out_features=out_phen_dim),
            nn.LeakyReLU(0.01),
        )

    def forward(self, x):
        x = self.decoder(x)
        return x


class PhenotypeDecoderCatGaussLatent(nn.Module):
    """A phenotype decoder that expects a latent space with
    dimensions that are categorical and other dimensions that are gaussian.
    It is intended to be used in a adversarial archetecture."""

    def __init__(self, phen_dim=None, N=None):
        if N is None:
            N = variables.d_hidden_dim
        if phen_dim is None:
            phen_dim = variables.n_phens
        n_env = variables.n_env
        n_allelic_states = variables.n_locs * variables.n_alleles
        latent_dim = n_allelic_states + n_env
        batchnorm_momentum = variables.batchnorm_momentum
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=N),
            nn.BatchNorm1d(N, momentum=batchnorm_momentum),
            nn.LeakyReLU(0.01),
            nn.Linear(in_features=N, out_features=phen_dim),
            nn.LeakyReLU(0.01),
        )

    def forward(self, cat_in, norm_in):
        input = torch.cat((cat_in, norm_in), -1)
        x = self.decoder(input)
        return x


class GenotypeDecoder(nn.Module):
    """A genotype decoder that outputs 1 hot encoded genotypes"""

    def __init__(self, n_loci=None, N=None):
        super().__init__()
        if n_loci is None:
            n_loci = variables.n_loci_measured * variables.n_alleles

        latent_dim = variables.latent_dim

        self.decoder = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=n_loci), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.decoder(x)
        return x


# discriminators
"""A series of discriminator networks that are intended for conditioning
the latent space of adversarial networks"""


class CategoricalGenotypeDiscriminator(nn.Module):
    """A categorical discriminator that conditions the latent space to be
    distributed according to the distribution of allelic states"""

    def __init__(self):
        super().__init__()

        n_allelic_states = variables.n_locs * variables.n_alleles
        n_hidden = variables.hidden_dim_dg

        self.discriminator = nn.Sequential(
            nn.Linear(in_features=n_allelic_states, out_features=n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.LeakyReLU(0.01),
            nn.Linear(in_features=n_hidden, out_features=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.discriminator(x)
        return x


class GaussianDiscriminator(nn.Module):
    """A discriminator that conditions the latent space to be normally
    distributed"""

    def __init__(self):
        super().__init__()

        n_env = variables.n_env
        n_hidden = variables.hidden_dim_de

        self.discriminator = nn.Sequential(
            nn.Linear(in_features=n_env, out_features=n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.LeakyReLU(0.01),
            nn.Linear(in_features=n_hidden, out_features=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.discriminator(x)
        return x
