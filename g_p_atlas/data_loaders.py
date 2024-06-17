import pickle as pk

import torch as torch
from torch.utils.data.dataset import Dataset


class simulated_phen_gen_dataset(Dataset):
    """a class for importing noisy simulated genotype-phenotype data.
    It expects a pickled object that is organized as a list of tensors:
    genotypes[n_animals, n_loci, n_alleles] (one hot at allelic state)
    gen_locs[n_animals, n_loci] (index of allelic state)
    weights[n_phens, n_loci, n_alleles] float weight for allelic contribution to phen
    phens[n_animals,n_phens] float value for phenotype
    indexes_of_loci_influencing_phen[n_phens,n_loci_ip] integer indicies of loci that influenc>
    interaction_matrix[FILL THIS IN]
    pleiotropy_matrix[n_phens, n_phens, gen_index]"""

    def __init__(self, data_file):
        self.datset = pk.load(open(data_file, "rb"))
        self.phens = self.datset["noisy_phens"]
        self.genotypes = self.datset["genotypes"]
        self.weights = self.datset["weights"]
        self.data_file = data_file

    def __len__(self):
        return len(self.phens)

    def __getitem__(self, idx):
        phenotypes = torch.tensor(self.phens[idx], dtype=torch.float32)
        genotype = torch.tensor(self.genotypes[idx], dtype=torch.float32)
        weights = torch.tensor(self.weights[idx], dtype=torch.fload32)
        return phenotypes, genotype, weights


class dgrp_phen_gen_dataset(Dataset):
    """a class for importing genotype-phenotype data from the DGRP.
    It expects a pickled object that is organized as a list of tensors:
    genotypes[n_animals, n_loci, n_alleles] (one hot at allelic state)
    gen_locs[n_animals, n_loci] (index of allelic state)
    weights[n_phens, n_loci, n_alleles] float weight for allelic contribution to phen
    phens[n_animals,n_phens] float value for phenotype
    indexes_of_loci_influencing_phen[n_phens,n_loci_ip] integer indicies of loci that influenc>
    interaction_matrix[FILL THIS IN]
    pleiotropy_matrix[n_phens, n_phens, gen_index]"""

    def __init__(self, data_file):
        self.datset = pk.load(open(data_file, "rb"))
        self.phens = self.datset["phenotypes"]
        self.genotypes = self.datset["genotypes"]
        self.data_file = data_file

    def __len__(self):
        return len(self.phens)

    def __getitem__(self, idx):
        phenotypes = torch.tensor(self.phens[idx], dtype=torch.float32)
        genotype = torch.tensor(self.genotypes[idx], dtype=torch.float32)
        return phenotypes, genotype
