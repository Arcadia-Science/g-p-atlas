import pathlib

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from gpatlas.models import GPAtlasModel, GPAtlasModelConfig
from gpatlas.trainers import GPAtlasTrainer, GPAtlasTrainerConfig


@pytest.fixture
def model() -> GPAtlasModel:
    """
    Create a model with small dimensions for fast testing.
    """
    # Use prime numbers for the model dimensions to avoid any incidental symmetries
    # that might mask bugs.
    model_config = GPAtlasModelConfig(
        num_loci=13,
        num_alleles=2,
        num_input_phenotypes=11,
        num_output_phenotypes=7,
        latent_dim=3,
        phenotype_encoder_hidden_dim=5,
        phenotype_decoder_hidden_dim=5,
        genotype_encoder_hidden_dim=5,
    )
    return GPAtlasModel(config=model_config)


@pytest.fixture
def trainer(tmp_path: pathlib.Path, model: GPAtlasModel) -> GPAtlasTrainer:
    """
    Create a trainer with a small number of epochs for fast testing.
    """
    checkpoint_filepath = tmp_path / "best_model_checkpoint.pt"
    trainer_config = GPAtlasTrainerConfig(
        seed=42,
        num_epochs=2,
        batch_size=4,
        learning_rate=0.01,
    )
    return GPAtlasTrainer(
        model=model, config=trainer_config, checkpoint_filepath=checkpoint_filepath
    )


@pytest.fixture
def train_dataloader(model: GPAtlasModel) -> DataLoader:
    """
    Create a dataloader to use for training.
    """
    n_samples = 9
    n_loci = model.config.num_loci
    n_alleles = model.config.num_alleles
    n_phenotypes = model.config.num_input_phenotypes

    # Generate random genotypes and phenotypes.
    genotypes = torch.randint(0, 2, (n_samples, n_loci * n_alleles), dtype=torch.float32)
    phenotypes = torch.rand((n_samples, n_phenotypes), dtype=torch.float32)

    dataset = TensorDataset(genotypes, phenotypes)
    dataloader = DataLoader(dataset, batch_size=3, shuffle=True)

    return dataloader


def test_trainer_initialization(trainer: GPAtlasTrainer):
    """Test that the trainer initializes correctly."""
    assert trainer.phenotype_encoder_optimizer is not None
    assert trainer.phenotype_decoder_optimizer is not None
    assert trainer.genotype_encoder_optimizer is not None
    assert trainer.phenotype_encoder_decoder_training_losses == []
    assert trainer.genotype_encoder_training_losses == []


def test_train(trainer: GPAtlasTrainer, train_dataloader: DataLoader):
    """Test the main training loop."""

    # Enable anomaly detection to raise errors if there are NaNs or Infs in the gradients.
    with torch.autograd.set_detect_anomaly(True, check_nan=True):
        trainer.train(train_dataloader)

    # Check that training losses were recorded.
    # There are 2 epochs of training, so there should be 2 training losses.
    assert len(trainer.genotype_encoder_training_losses) == 2
    assert len(trainer.phenotype_encoder_decoder_training_losses) == 2


def test_predict_phenotypes(trainer: GPAtlasTrainer, train_dataloader: DataLoader):
    """Test the phenotype prediction process."""

    num_samples = len(train_dataloader.dataset)  # type: ignore
    trainer.train(train_dataloader)

    # Create a test dataloader with a single batch for prediction.
    test_dataloader = DataLoader(
        train_dataloader.dataset, batch_size=trainer.config.batch_size, shuffle=False
    )

    # Test prediction with phenotype encoder.
    true_phenotypes, decoded_phenotypes, embeddings = trainer.predict_phenotypes(
        loader=test_dataloader, input_type="phenotype"
    )

    # Check output shapes.
    assert true_phenotypes.shape == (num_samples, trainer.model.config.num_input_phenotypes)
    assert decoded_phenotypes.shape == (num_samples, trainer.model.config.num_output_phenotypes)
    assert embeddings.shape == (num_samples, trainer.model.config.latent_dim)

    # Test prediction with genotype encoder.
    true_phenotypes, decoded_phenotypes, embeddings = trainer.predict_phenotypes(
        loader=test_dataloader, input_type="genotype"
    )

    # Check output shapes.
    assert true_phenotypes.shape == (num_samples, trainer.model.config.num_input_phenotypes)
    assert decoded_phenotypes.shape == (num_samples, trainer.model.config.num_output_phenotypes)
    assert embeddings.shape == (num_samples, trainer.model.config.latent_dim)


def test_calculate_feature_importances(trainer: GPAtlasTrainer, train_dataloader: DataLoader):
    """Test the feature importance calculation."""

    num_samples = len(train_dataloader.dataset)  # type: ignore
    trainer.train(train_dataloader)

    test_dataloader = DataLoader(
        train_dataloader.dataset, batch_size=trainer.config.batch_size, shuffle=False
    )

    importances = trainer.calculate_feature_importances(test_dataloader, input_type="genotype")
    assert importances.shape == (
        num_samples,
        trainer.model.config.num_loci * trainer.model.config.num_alleles,
    )

    importances = trainer.calculate_feature_importances(test_dataloader, input_type="phenotype")
    assert importances.shape == (num_samples, trainer.model.config.num_input_phenotypes)
