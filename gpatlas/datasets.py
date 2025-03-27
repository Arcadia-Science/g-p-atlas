import pickle

import numpy as np
import torch
from torch.utils.data.dataset import Dataset


class GPAtlasDataset(Dataset):
    """
    A class for importing simulated genotype and phenotype data.

    It expects a pickled object that is organized as a dictionary of arrays:
    'genotypes': array of shape [n_animals, n_loci, n_alleles] (one hot at allelic state)
    'phenotypes': array of shape [n_animals, n_phens] (float values of the phenotypes)
    """

    def __init__(self, filepath: str, num_phenotypes: int | None = None):
        """
        Initialize a GPAtlasDataset.

        Note: `num_phenotypes` was set to `args.n_phens_to_analyze` in the original implementation.
        """

        # TODO: use hdf5 files instead of pickle files.
        with open(filepath, "rb") as f:
            self.dataset = pickle.load(f)

        self.phenotypes = torch.tensor(np.array(self.dataset["phenotypes"]), dtype=torch.float32)
        self.genotypes = torch.tensor(np.array(self.dataset["genotypes"]), dtype=torch.float32)

        if len(self.phenotypes) != len(self.genotypes):
            raise ValueError("The number of phenotypes and genotypes must be the same.")

        # Crop the phenotypes to the specified number.
        # TODO: should we randomize the order of the phenotypes here?
        if num_phenotypes is not None:
            self.phenotypes = self.phenotypes[:, :num_phenotypes]

    def __len__(self) -> int:
        return len(self.genotypes)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        genotype = torch.flatten(self.genotypes[idx])
        phenotypes = self.phenotypes[idx]
        return genotype, phenotypes
