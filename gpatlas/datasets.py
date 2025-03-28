import h5py
import numpy as np
import torch
from torch.utils.data.dataset import Dataset


class GPAtlasDataset(Dataset):
    def __init__(self, filepath: str, num_phenotypes: int | None = None):
        """
        Initialize a GPAtlasDataset.

        filepath: the path to an hdf5 file containing the data.
            The file should contain the following datasets:
            - 'genotypes': array of one-hot encoded genotypes (n_samples, n_loci, n_alleles)
            - 'phenotypes': array of phenotype values (n_samples, n_phenotypes)

        num_phenotypes: the number of phenotypes to use. If `None`, all phenotypes will be used.

        Note: `num_phenotypes` was set to `args.n_phens_to_analyze` in the original implementation.
        """

        with h5py.File(filepath, "r") as file:
            self.phenotypes = torch.tensor(np.array(file["phenotypes"]), dtype=torch.float32)
            self.genotypes = torch.tensor(np.array(file["genotypes"]), dtype=torch.float32)

        if self.phenotypes.shape[0] != self.genotypes.shape[0]:
            raise ValueError("The number of phenotypes and genotypes must be the same.")

        # Crop the phenotypes to the specified number.
        # TODO: should we randomize the order of the phenotypes here?
        if num_phenotypes is not None:
            self.phenotypes = self.phenotypes[:, :num_phenotypes]

    def __len__(self) -> int:
        return self.genotypes.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        genotype = torch.flatten(self.genotypes[idx])
        phenotypes = self.phenotypes[idx]
        return genotype, phenotypes
