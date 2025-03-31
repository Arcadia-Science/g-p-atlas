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
    model_config = GPAtlasModelConfig(
        num_loci_measured=10,
        num_input_phenotypes=5,
        num_output_phenotypes=5,
        latent_dim=4,
        phenotype_encoder_hidden_dim=8,
        phenotype_decoder_hidden_dim=8,
        genotype_encoder_hidden_dim=8,
    )
    return GPAtlasModel(config=model_config)


@pytest.fixture
def trainer(model: GPAtlasModel) -> GPAtlasTrainer:
    """
    Create a trainer with a small number of epochs for fast testing.
    """
    trainer_config = GPAtlasTrainerConfig(
        seed=42,
        num_epochs=2,
        batch_size=4,
        learning_rate=0.01,
    )
    return GPAtlasTrainer(model=model, config=trainer_config)


@pytest.fixture
def train_dataloader() -> DataLoader:
    """
    Create a dataloader to use for training.
    """
    n_samples = 16
    n_loci = 10
    n_alleles = 2
    n_phenotypes = 5

    # Generate random genotypes and phenotypes.
    genotypes = torch.randint(0, 2, (n_samples, n_loci * n_alleles), dtype=torch.float32)
    phenotypes = torch.rand((n_samples, n_phenotypes), dtype=torch.float32)

    dataset = TensorDataset(genotypes, phenotypes)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

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

    trainer.train(train_dataloader)

    # Check that training losses were recorded.
    # There are 4 batches in the dataloader, and 2 epochs of training,
    # so there should be 8 training losses.
    assert len(trainer.genotype_encoder_training_losses) == 8


def test_predict_phenotypes(trainer: GPAtlasTrainer, train_dataloader: DataLoader):
    """Test the phenotype prediction process."""

    trainer.train(train_dataloader)

    # Create a test dataloader with a single batch for prediction.
    test_dataloader = DataLoader(train_dataloader.dataset, batch_size=1, shuffle=False)

    # Test prediction with phenotype encoder.
    true_phenotypes, decoded_phenotypes, embeddings = trainer.predict_phenotypes(
        loader=test_dataloader, input_type="phenotype"
    )

    # Check output shapes.
    assert true_phenotypes.shape == (
        len(test_dataloader),
        trainer.model.config.num_input_phenotypes,
    )
    assert decoded_phenotypes.shape == (
        len(test_dataloader),
        trainer.model.config.num_output_phenotypes,
    )
    assert embeddings.shape == (len(test_dataloader), trainer.model.config.latent_dim)

    # Test prediction with genotype encoder.
    true_phenotypes, decoded_phenotypes, embeddings = trainer.predict_phenotypes(
        loader=test_dataloader, input_type="genotype"
    )

    # Check output shapes.
    assert true_phenotypes.shape == (
        len(test_dataloader),
        trainer.model.config.num_input_phenotypes,
    )
    assert decoded_phenotypes.shape == (
        len(test_dataloader),
        trainer.model.config.num_output_phenotypes,
    )
    assert embeddings.shape == (len(test_dataloader), trainer.model.config.latent_dim)


def test_calculate_feature_importances(trainer: GPAtlasTrainer, train_dataloader: DataLoader):
    """Test the feature importance calculation."""

    trainer.train(train_dataloader)

    importances = trainer.calculate_feature_importances(train_dataloader, input_type="genotype")
    assert importances.shape == (
        len(train_dataloader),
        trainer.model.config.num_loci_measured * trainer.model.config.num_alleles,
    )

    importances = trainer.calculate_feature_importances(train_dataloader, input_type="phenotype")
    assert importances.shape == (
        len(train_dataloader),
        trainer.model.config.num_output_phenotypes,
    )
