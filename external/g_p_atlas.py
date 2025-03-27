import pickle as pk
import sys
import time as tm
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
from captum.attr import FeatureAblation
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data.dataset import Dataset

"""This is an implementation of the G-P Atlas method for mapping genotype
to phenotype described in https://doi.org/10.57844/arcadia-d316-721f.
For help type:
python3 g_p_atlas.py --help"""

# parse commandline arguments
args = ArgumentParser()
args.add_argument(
    "--n_alleles",
    type=int,
    default=2,
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
    "--gen_lw", type=float, default=1, help="weight for the loss attributed to genetic features"
)
args.add_argument(
    "--eng_lw", type=float, default=0.1, help="weight for the loss attributed to env features"
)
args.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
args.add_argument("--batch_size", type=int, default=16, help="batch size")
args.add_argument("--lr_r", type=float, default=0.001, help="reconstruction learning rate")
args.add_argument("--b1", type=float, default=0.5, help="adam: gradient decay variables")
args.add_argument("--b2", type=float, default=0.999, help="adam: gradient decay variables")
args.add_argument("--n_cpu", type=int, default=14, help="number of cpus")
args.add_argument(
    "--e_hidden_dim", type=int, default=32, help="number of neurons in the hidden layers of encoder"
)
args.add_argument(
    "--d_hidden_dim", type=int, default=32, help="number of neurons in the hidden layers of decoder"
)
args.add_argument(
    "--ge_hidden_dim",
    type=int,
    default=32,
    help="number of neurons in the hidden layers of the genetic encoder",
)
args.add_argument(
    "--batchnorm_momentum",
    type=float,
    default=0.8,
    help="momentum for the batch normalization layers",
)
args.add_argument(
    "--latent_dim", type=int, default=32, help="number of neurons in the latent space"
)
args.add_argument(
    "--n_phens_to_analyze", type=int, default=30, help="number of phenotypes to analyze"
)
args.add_argument("--sd_noise", type=float, default=0.1, help="noise added to phens")
args.add_argument("--gen_noise", type=float, default=0.3, help="noise added to gens")
args.add_argument(
    "--n_phens_to_predict", type=int, default=30, help="number of phenotypes to predict"
)
args.add_argument(
    "--n_epochs_gen", type=int, default=100, help="number of epochs to train the genetic encoder"
)
args.add_argument(
    "--n_loci_measured",
    type=int,
    default=3000,
    help="number of genetic loci for which there is data",
)
args.add_argument(
    "--l1_lambda", type=float, default=0.8, help="l1 regularization weight for the genetic weights"
)
args.add_argument(
    "--l2_lambda",
    type=float,
    default=0.01,
    help="l2 regularization weight for the genetic weights",
)
args.add_argument(
    "--dataset_path", type=str, default=None, help="where the train and test files are located"
)
args.add_argument(
    "--train_suffix",
    type=str,
    default="train.pk",
    help="name of the training data file, defaults to DGRP dataset",
)
args.add_argument("--test_suffix", type=str, default="test.pk", help="name of the test data file")
args.add_argument(
    "--hot_start",
    type=bool,
    default=False,
    help="flag to use precomputed weights, false by default",
)
args.add_argument(
    "--hot_start_path_e",
    type=str,
    default=None,
    help="path to the phenotype encoder weights to use at initialization",
)
args.add_argument(
    "--hot_start_path_d",
    type=str,
    default=None,
    help="path to the phenotype decoder weights to use at initialization",
)
args.add_argument(
    "--hot_start_path_ge",
    type=str,
    default=None,
    help="path to the genotype encoder weights to use at initialization",
)
args.add_argument(
    "--calculate_importance",
    type=str,
    default="no",
    help="flag whether to calculate variable importance. Expects 'no' or 'yes.' ",
)

vabs = args.parse_args()


# define a torch dataset object
class dataset_pheno(Dataset):
    """a class for importing simulated phenotype data.
    It expects a pickled object that is organized as a dictionary of tensors:
    phenotypes[n_animals,n_phens] float value for phenotype
    """

    def __init__(self, data_file, n_phens):
        self.datset = pk.load(open(data_file, "rb"))
        self.phens = torch.tensor(np.array(self.datset["phenotypes"]), dtype=torch.float32)
        self.data_file = data_file
        self.n_phens = n_phens

    def __len__(self):
        return len(self.phens)

    def __getitem__(self, idx):
        phenotypes = self.phens[idx][: self.n_phens]
        return phenotypes


class dataset_geno(Dataset):
    """a class for importing simulated genotype and phenotype data.
    It expects a pickled object that is organized as a dictionary of tensors:
    genotypes[n_animals, n_loci, n_alleles] (one hot at allelic state)
    phenotypes[n_animals,n_phens] float value for phenotype
    """

    def __init__(self, data_file, n_geno, n_phens):
        self.datset = pk.load(open(data_file, "rb"))
        self.phens = torch.tensor(np.array(self.datset["phenotypes"]), dtype=torch.float32)
        self.genotypes = torch.tensor(np.array(self.datset["genotypes"]), dtype=torch.float32)
        self.data_file = data_file
        self.n_geno = n_geno
        self.n_phens = n_phens

    def __len__(self):
        return len(self.genotypes)

    def __getitem__(self, idx):
        phenotypes = self.phens[idx][: self.n_phens]
        genotype = torch.flatten(self.genotypes[idx])
        return phenotypes, genotype


# helper functions
def sequential_forward_attr_gen_phen(input, phens):
    """puts together two models for use of captum feature
    importance in genotype-phenotype prediction"""
    mod_2_input = GQ(input)
    X_sample = P(mod_2_input)
    output = F.mse_loss(X_sample + EPS, phens[:, :n_phens_pred] + EPS)
    return output


def sequential_forward_attr_phen_phen(input, phens):
    """puts together two models for use of captum feature
    importance in phenotype-phenotype prediction"""
    mod_2_input = Q(input)
    X_sample = P(mod_2_input)
    output = F.mse_loss(X_sample + EPS, phens[:, :n_phens_pred] + EPS)
    return output


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true + EPS), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# load the training and test datasets
dataset_path = vabs.dataset_path

params_file = open(dataset_path + "run_params.txt", "w")
params_file.write(" ".join(sys.argv[:]))
params_file.close()

train_dat = vabs.train_suffix
test_dat = vabs.test_suffix

train_data_pheno = dataset_pheno(dataset_path + train_dat, n_phens=vabs.n_phens_to_analyze)
test_data_pheno = dataset_pheno(dataset_path + test_dat, n_phens=vabs.n_phens_to_analyze)

train_data_geno = dataset_geno(
    dataset_path + train_dat, n_geno=vabs.n_loci_measured, n_phens=vabs.n_phens_to_analyze
)
test_data_geno = dataset_geno(
    dataset_path + test_dat, n_geno=vabs.n_loci_measured, n_phens=vabs.n_phens_to_analyze
)

# setting device on GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# how many samples per batch to load
batch_size = vabs.batch_size
num_workers = vabs.n_cpu

# prepare data loaders
train_loader_pheno = torch.utils.data.DataLoader(
    dataset=train_data_pheno, batch_size=batch_size, num_workers=num_workers, shuffle=True
)
test_loader_pheno = torch.utils.data.DataLoader(
    dataset=test_data_pheno, batch_size=1, num_workers=num_workers, shuffle=True
)

train_loader_geno = torch.utils.data.DataLoader(
    dataset=train_data_geno, batch_size=batch_size, num_workers=num_workers, shuffle=True
)
test_loader_geno = torch.utils.data.DataLoader(
    dataset=test_data_geno, batch_size=1, num_workers=num_workers, shuffle=True
)


# model (encoders and decoders) classes
# phenotype encoder class
class Q_net(nn.Module):
    """
    Encoder for creating embeddings of phenotypic data.
    Parameters:
        out_phen_dim (int): Number of output phenotypes.
        N (int): Number of channels in hidden layers.
    """

    def __init__(self, phen_dim=None, N=None):
        super().__init__()
        if N is None:
            N = vabs.e_hidden_dim
        if phen_dim is None:
            phen_dim = vabs.n_phens_to_analyze

        batchnorm_momentum = vabs.batchnorm_momentum
        latent_dim = vabs.latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(in_features=phen_dim, out_features=N),
            nn.BatchNorm1d(N, momentum=batchnorm_momentum),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(in_features=N, out_features=latent_dim),
            nn.BatchNorm1d(latent_dim, momentum=batchnorm_momentum),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x):
        x = self.encoder(x)
        return x


# phenotype decoder class
class P_net(nn.Module):
    """
    Decoder for predicting phenotypes from either genotypic or phenotypic data.
    Parameters:
        out_phen_dim (int): Number of output phenotypes.
        N (int): Number of channels in hidden layers.
    """

    def __init__(self, out_phen_dim=None, N=None):
        if N is None:
            N = vabs.d_hidden_dim
        if out_phen_dim is None:
            out_phen_dim = vabs.n_phens_to_predict

        latent_dim = vabs.latent_dim
        batchnorm_momentum = vabs.batchnorm_momentum

        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=N),
            nn.BatchNorm1d(N, momentum=batchnorm_momentum),
            nn.LeakyReLU(0.01),
            nn.Linear(in_features=N, out_features=out_phen_dim),
        )

    def forward(self, x):
        x = self.decoder(x)
        return x


# genetic encoder class
class GQ_net(nn.Module):
    """
    Genetic encoder to produce latent embedding of genotypic data for predicting
    either phenotypes or genotypes.

    Parameters:
        n_loci (int): number of input measured loci * number of segregating alleles
        N (int): Number of channels in hidden layers.
    """

    def __init__(self, n_loci=None, N=None):
        super().__init__()
        if N is None:
            N = vabs.ge_hidden_dim
        if n_loci is None:
            n_loci = vabs.n_loci_measured * vabs.n_alleles

        batchnorm_momentum = vabs.batchnorm_momentum
        latent_dim = vabs.latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(in_features=n_loci, out_features=N),
            nn.BatchNorm1d(N, momentum=batchnorm_momentum),
            nn.LeakyReLU(0.01),
            nn.Linear(in_features=N, out_features=latent_dim),
            nn.BatchNorm1d(latent_dim, momentum=batchnorm_momentum),
            nn.LeakyReLU(0.01),
        )

    def forward(self, x):
        x = self.encoder(x)
        return x


# G-P Atlas run

EPS = 1e-15  # define a minimum value for variables to avoid divide by zero and buffer underflow

# define encoders and decoders
Q = Q_net()  # phenotype encoder
P = P_net()  # phenotype decoder
GQ = GQ_net()  # genotype encoder

# load precomputed weights
# this allows you to specify precomputed weights so that the training can be 'hotstarted'
if vabs.hot_start is True:
    Q.load_state_dict(torch.load(vabs.hot_start_path_e, weights_only=True), strict=False)
    P.load_state_dict(torch.load(vabs.hot_start_path_d, weights_only=True), strict=False)
    GQ.load_state_dict(torch.load(vabs.hot_start_path_ge, weights_only=True), strict=False)

# put the models on the GPU if it is there
Q.to(device)
P.to(device)
GQ.to(device)

# Set up feature importance measure
fa = FeatureAblation(sequential_forward_attr_gen_phen)  # genotype feature importance
fa_p = FeatureAblation(sequential_forward_attr_phen_phen)  # phenotype feature importance

# Set learning rates
reg_lr = vabs.lr_r

# adam betas
adam_b = (vabs.b1, vabs.b2)

# encode/decode optimizers
optim_P = torch.optim.Adam(P.parameters(), lr=reg_lr, betas=adam_b)
optim_Q_enc = torch.optim.Adam(Q.parameters(), lr=reg_lr, betas=adam_b)
optim_GQ_enc = torch.optim.Adam(GQ.parameters(), lr=reg_lr, betas=adam_b)

# set the number of training epochs for the phenotype-phenotype autoencoder
num_epochs = vabs.n_epochs

torch.manual_seed(47)

# establish the number of phenotypes to predict and the number to use for prediction.
n_phens = vabs.n_phens_to_analyze
n_phens_pred = vabs.n_phens_to_predict

# establish a variable to capture the reconstruction loss
rcon_loss = []

# establish a variable for the start of the run
start_time = tm.time()

# train the phenotype encoder and decoder
for n in range(num_epochs):
    for _i, (phens) in enumerate(train_loader_pheno):
        phens = phens[:, :n_phens]  # constrain the number of phenotypes to use for prediction
        phens = phens.to(device)  # move data to GPU if it is there
        batch_size = phens.shape[0]  # redefine batch size here to allow for incomplete batches

        P.zero_grad()  # initialize gradients for training
        Q.zero_grad()

        noise_phens = phens + (vabs.sd_noise**0.5) * torch.randn(phens.shape).to(device)
        # add noise to phenotypes

        z_sample = Q(noise_phens)  # encode phenotypes
        X_sample = P(z_sample)  # decode encodings to produce predicted phenotypes

        recon_loss = F.mse_loss(X_sample + EPS, phens[:, :n_phens_pred] + EPS)
        # calculate the error of the phenotype predicitons

        rcon_loss.append(float(recon_loss.detach()))  # add the loss to the aggregator

        recon_loss.backward()  # back propagate the reconstruction loss through the autoencoder
        optim_P.step()  # step the optimizers
        optim_Q_enc.step()

    cur_time = tm.time() - start_time  # calculate the time it took for this batch
    start_time = tm.time()  # re-initialize the start time
    # for each loop, print a set of useful information
    print(
        "Epoch num: "
        + str(n)
        + " batchno "
        + str(_i)
        + " r_con_loss: "
        + str(rcon_loss[-1])
        + " epoch duration: "
        + str(cur_time)
    )


# train genetic network

P.requires_grad_(False)  # freeze weights in P (decoder)
P.eval()  # put P (phenotype decoder) into evaluation mode
num_epochs_gen = vabs.n_epochs_gen

gen_noise = 1 - vabs.gen_noise

g_rcon_loss = []  # establish a variable to contain the reconstruction loss values

start_time = tm.time()

for n in range(num_epochs_gen):
    for _i, (phens, gens) in enumerate(train_loader_geno):
        phens = phens.to(device)  # move phenotypic data to the gpu if it is there

        gens = gens[:, : vabs.n_loci_measured * vabs.n_alleles]

        pos_noise = np.random.binomial(1, gen_noise / 2, gens.shape)

        neg_noise = np.random.binomial(1, gen_noise / 2, gens.shape)

        noise_gens = torch.tensor(
            np.where((gens + pos_noise - neg_noise) > 0, 1, 0), dtype=torch.float32
        )  # add noise to the genetic data

        noise_gens = noise_gens.to(device)  # put genotypes plus noise on the gpu if it is there

        batch_size = phens.shape[0]  # establish the training batch size

        GQ.zero_grad()  # zero the gradients

        z_sample = GQ(noise_gens)  # encode the genetic data
        X_sample = P(z_sample)  # decode the encoded genetic data to phenotypes

        g_recon_loss = F.mse_loss(X_sample + EPS, phens[:, :n_phens_pred] + EPS)

        g_rcon_loss.append(float(g_recon_loss.detach()))

        # Calculate the L1 and L2 norms for the weights in the first layer of the
        # genetic encoder and add them to the reconstruction loss
        l1_reg = torch.linalg.norm(torch.sum(GQ.encoder[0].weight, axis=0), 1)
        l2_reg = torch.linalg.norm(torch.sum(GQ.encoder[0].weight, axis=0), 2)
        g_recon_loss = g_recon_loss + l1_reg * vabs.l1_lambda + l2_reg * vabs.l2_lambda

        g_recon_loss.backward()  # backpropagate the loss

        optim_GQ_enc.step()  # step the optimizer

    cur_time = tm.time() - start_time  # set a variable for the epoch time
    start_time = tm.time()  # set the original start time variable to the current time

    # print useful things about the current training epoch
    print(
        "Epoch num: "
        + str(n)
        + " batchno "
        + str(_i)
        + " r_con_loss: "
        + str(g_rcon_loss[-1])
        + " epoch duration: "
        + str(cur_time)
    )

# plot the reconstruction losses
plt.plot(rcon_loss)  # reconstruction loss for the phenotype autoencoder
plt.plot(g_rcon_loss)  # reconstruction loss for the genetic weights
plt.savefig(dataset_path + "reconstruction_loss.svg")
plt.close()


# A function to evaluate the performance of each model, saving summaries of model performance
def analyze_predictions(
    phens,
    phen_encodings,
    phen_latent,
    fa_attr,
    dataset_path,
    n_phens_pred,
    model_type="g_p",
):
    """Analyze predictions and save visualization results.

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
        pk.dump(fa_attr, open(dataset_path + f"{model_type}_attr.pk", "wb"))

    # Convert and transpose data
    phens = np.array(phens).T
    phen_encodings = np.array(phen_encodings).T

    # Save predictions data
    if model_type == "p_p":
        phen_latent = np.array(phen_latent).T
        pk.dump(
            [phens, phen_encodings, phen_latent],
            open(dataset_path + f"phens_phen_encodings_dng_attr{suffix}.pk", "wb"),
        )
    else:
        pk.dump(
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


# Now use this function to analyze predictions and save outputs
out_stats = open(dataset_path + "test_stats.pk", "wb")
stats_aggregator = []

# Save model states first
torch.save(Q.state_dict(), dataset_path + "phen_encoder_state.pt")
torch.save(P.state_dict(), dataset_path + "phen_decoder_state.pt")
torch.save(GQ.state_dict(), dataset_path + "gen_encoder_state.pt")

# G-P prediction. Loop evaluating the performance on the test data
GQ.eval()
phens, phen_encodings, phen_latent, fa_attr = [], [], [], []

for dat in test_loader_geno:
    ph, gt = dat
    gt = gt[:, : vabs.n_loci_measured * vabs.n_alleles]
    ph = ph.to(device)
    gt = gt.to(device)
    batch_size = ph.shape[0]
    z_sample = GQ(gt)
    X_sample = P(z_sample)
    phens += list(ph.detach().cpu().numpy())
    phen_encodings += list(X_sample.detach().cpu().numpy())
    phen_latent += list(z_sample.detach().cpu().numpy())
    if vabs.calculate_importance == "yes":
        fa_attr.append(list(fa.attribute(inputs=(gt, ph))[0].squeeze().detach().cpu().numpy()))

stats_aggregator.extend(
    analyze_predictions(
        phens, phen_encodings, phen_latent, fa_attr, dataset_path, n_phens_pred, "g_p"
    )
)

# P-P prediction. Loop evaluating the performance on the test data
Q.eval()
phens, phen_encodings, phen_latent, fa_attr = [], [], [], []

for dat in test_loader_pheno:
    ph = dat
    ph = ph.to(device)
    batch_size = ph.shape[0]
    z_sample = Q(ph)
    X_sample = P(z_sample)
    phens += list(ph.detach().cpu().numpy())
    phen_encodings += list(X_sample.detach().cpu().numpy())
    phen_latent += list(z_sample.detach().cpu().numpy())
    if vabs.calculate_importance == "yes":
        fa_attr.append(list(fa_p.attribute(inputs=(ph, ph))[0].squeeze().detach().cpu().numpy()))

stats_aggregator.extend(
    analyze_predictions(
        phens, phen_encodings, phen_latent, fa_attr, dataset_path, n_phens_pred, "p_p"
    )
)

# Save and close stats
pk.dump(stats_aggregator, out_stats)
out_stats.close()
