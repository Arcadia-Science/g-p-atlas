import h5py
import numpy as np
import torch
from torch.utils.data.dataset import Dataset

EPS = 1e-8


class GPAtlasDataset(Dataset):
    def __init__(self, filepath: str, num_phenotypes: int | None = None):
        """
        Initialize a GPAtlasDataset.

        filepath: the path to an hdf5 file containing the following datasets:
            - 'genotypes': array of one-hot encoded allelic states with shape
                `(n_samples, n_loci, n_alleles)`.
            - 'phenotypes': array of phenotype values with shape `(n_samples, n_phenotypes)`.

        num_phenotypes: the number of phenotypes to use. If `None`, all phenotypes will be used.

        Note: `num_phenotypes` was set to `args.n_phens_to_analyze` in the original implementation.
        """

        with h5py.File(filepath, "r") as file:
            if "phenotypes" in file and "genotypes" in file:
                self.phenotypes = torch.tensor(np.array(file["phenotypes"]), dtype=torch.float32)
                self.genotypes = torch.tensor(np.array(file["genotypes"]), dtype=torch.float32)
            elif "arrays" in file:
                self.phenotypes = torch.tensor(
                    np.array(file["arrays/phenotypes"]), dtype=torch.float32
                )
                self.genotypes = torch.tensor(
                    np.array(file["arrays/genotypes"]), dtype=torch.float32
                )
            else:
                raise ValueError("The dataset must contain either phenotypes or genotypes.")

        if self.phenotypes.shape[0] != self.genotypes.shape[0]:
            raise ValueError("The number of phenotypes and genotypes must be the same.")

        # Take the first `num_phenotypes` phenotypes.
        if num_phenotypes is not None:
            self.phenotypes = self.phenotypes[:, :num_phenotypes]

        # Mean-center and scale the phenotypes.
        self.phenotypes = (self.phenotypes - self.phenotypes.mean(dim=0)) / (
            self.phenotypes.std(dim=0) + EPS
        )

    def __len__(self) -> int:
        return self.genotypes.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Flatten the 2nd and 3rd dimensions of `genotypes`, as the model expects a single
        # input dimension of one-hot encoded alleles for the genotype.
        genotype = torch.flatten(self.genotypes[idx])
        phenotypes = self.phenotypes[idx]
        return genotype, phenotypes
