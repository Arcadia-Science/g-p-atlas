import pathlib

import h5py
import numpy as np
import pytest
import torch

from gpatlas.datasets import GPAtlasDataset


@pytest.fixture
def mock_h5_dataset(tmp_path: pathlib.Path):
    """
    Create a temporary HDF5 file with mock data for testing.
    """
    n_samples = 10
    n_loci = 50
    n_alleles = 2
    n_phenotypes = 20

    # Generate random test data.
    genotypes = np.random.randint(0, 2, size=(n_samples, n_loci, n_alleles)).astype(np.float32)
    phenotypes = np.random.rand(n_samples, n_phenotypes).astype(np.float32)

    filepath = tmp_path / "dataset.h5"
    with h5py.File(filepath, "w") as f:
        f.create_dataset("genotypes", data=genotypes)
        f.create_dataset("phenotypes", data=phenotypes)

    # Return the path and expected shapes.
    return (
        filepath,
        {
            "n_samples": n_samples,
            "n_loci": n_loci,
            "n_alleles": n_alleles,
            "n_phenotypes": n_phenotypes,
        },
    )


def test_dataset_full_phenotypes(mock_h5_dataset):
    """
    Test loading the dataset with all phenotypes.
    """
    filepath, shapes = mock_h5_dataset
    dataset = GPAtlasDataset(filepath=filepath)

    # Check shapes.
    assert len(dataset) == shapes["n_samples"]
    assert dataset.phenotypes.shape == (shapes["n_samples"], shapes["n_phenotypes"])
    assert dataset.genotypes.shape == (shapes["n_samples"], shapes["n_loci"], shapes["n_alleles"])

    # Check item retrieval.
    genotype, phenotype = dataset[0]
    assert genotype.shape == (shapes["n_loci"] * shapes["n_alleles"],)
    assert phenotype.shape == (shapes["n_phenotypes"],)

    # Check data types.
    assert genotype.dtype == torch.float32
    assert phenotype.dtype == torch.float32


def test_dataset_flattened_genotypes(mock_h5_dataset):
    """
    Test that genotypes are properly flattened when retrieved.
    """
    filepath, shapes = mock_h5_dataset
    dataset = GPAtlasDataset(filepath=filepath)

    # Get a batch of data.
    genotype, _ = dataset[0]

    # Check that the genotype is flattened.
    assert genotype.shape == (shapes["n_loci"] * shapes["n_alleles"],)

    # Verify flattening is correct by comparing with manual flattening.
    original_genotype = dataset.genotypes[0]
    manual_flatten = torch.flatten(original_genotype)
    assert torch.allclose(genotype, manual_flatten)
