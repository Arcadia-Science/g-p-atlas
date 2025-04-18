import pathlib

import pytest

from gpatlas.models import GPAtlasModel, GPAtlasModelConfig


def test_model_initialization():
    """
    Test that the model initializes correctly.
    """
    config = GPAtlasModelConfig(
        num_loci=1000, num_input_phenotypes=20, num_output_phenotypes=20, latent_dim=16
    )

    model = GPAtlasModel(config=config)

    # Check that all components were initialized.
    assert model.phenotype_encoder is not None
    assert model.phenotype_decoder is not None
    assert model.genotype_encoder is not None

    # Check input/output dimensions.
    assert model.phenotype_encoder.layers[0].in_features == config.num_input_phenotypes
    assert model.phenotype_decoder.layers[-1].out_features == config.num_output_phenotypes
    assert model.genotype_encoder.layers[0].in_features == (config.num_loci * config.num_alleles)


def test_model_freeze_unfreeze():
    """
    Test that the model components can be frozen and unfrozen.
    """
    config = GPAtlasModelConfig(
        num_loci=100, num_input_phenotypes=10, num_output_phenotypes=10, latent_dim=8
    )

    model = GPAtlasModel(config=config)

    # By default, all parameters should require gradients
    for param in model.phenotype_encoder.parameters():
        assert param.requires_grad
    for param in model.phenotype_decoder.parameters():
        assert param.requires_grad
    for param in model.genotype_encoder.parameters():
        assert param.requires_grad

    # Freeze phenotype encoder and decoder.
    model.freeze_phenotype_encoder_decoder()

    # Check that phenotype components are frozen but genotype encoder is not
    for param in model.phenotype_encoder.parameters():
        assert not param.requires_grad
    for param in model.phenotype_decoder.parameters():
        assert not param.requires_grad
    for param in model.genotype_encoder.parameters():
        assert param.requires_grad

    # Freeze genotype encoder.
    model.freeze_genotype_encoder()

    # Check that all components are now frozen.
    for param in model.genotype_encoder.parameters():
        assert not param.requires_grad


def test_model_checkpoint(tmp_path: pathlib.Path):
    """
    Test that the model can be saved to and loaded from a checkpoint.
    """
    config = GPAtlasModelConfig(
        num_loci=100, num_input_phenotypes=10, num_output_phenotypes=10, latent_dim=8
    )

    model = GPAtlasModel(config=config)

    checkpoint_path = tmp_path / "model.pt"
    model.save_to_checkpoint(checkpoint_path)

    # Try to load the model from the checkpoint.
    model.load_from_checkpoint(checkpoint_path)

    # Initialize a new model with a different config.
    new_config = config.model_copy(update={"num_loci": config.num_loci * 2})
    new_model = GPAtlasModel(config=new_config)

    # Load the new model from the checkpoint. This should raise an error, because the model config
    # is different.
    with pytest.raises(RuntimeError):
        new_model.load_from_checkpoint(checkpoint_path)
