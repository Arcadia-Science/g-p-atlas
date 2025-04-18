import torch

from gpatlas.layers import Decoder, Encoder


def test_encoder_forward():
    """
    Test that the encoder produces expected output shapes.
    """
    batch_size = 16
    input_dim = 100
    hidden_dim = 50
    latent_dim = 32
    batchnorm_momentum = 0.8

    # Create a random input tensor.
    x = torch.randn(batch_size, input_dim)

    encoder = Encoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        batchnorm_momentum=batchnorm_momentum,
    )

    # Test forward pass.
    output = encoder(x)
    assert output.shape == (batch_size, latent_dim)

    # Test encoder without hidden layer.
    encoder_no_hidden = Encoder(
        input_dim=input_dim,
        hidden_dim=None,
        latent_dim=latent_dim,
        batchnorm_momentum=batchnorm_momentum,
    )

    # Test forward pass.
    output_no_hidden = encoder_no_hidden(x)
    assert output_no_hidden.shape == (batch_size, latent_dim)


def test_decoder_forward():
    """
    Test that the decoder produces expected output shapes.
    """
    batch_size = 16
    latent_dim = 32
    hidden_dim = 50
    output_dim = 100
    batchnorm_momentum = 0.8

    # Create a random input tensor.
    x = torch.randn(batch_size, latent_dim)

    # Create a decoder with a hidden layer.
    decoder = Decoder(
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        batchnorm_momentum=batchnorm_momentum,
    )

    # Test forward pass.
    output = decoder(x)
    assert output.shape == (batch_size, output_dim)

    # Test decoder without hidden layer.
    decoder_no_hidden = Decoder(
        latent_dim=latent_dim,
        hidden_dim=None,
        output_dim=output_dim,
        batchnorm_momentum=batchnorm_momentum,
    )

    # Test forward pass.
    output_no_hidden = decoder_no_hidden(x)
    assert output_no_hidden.shape == (batch_size, output_dim)


def test_encoder_to_decoder():
    """
    Test the encoder-decoder together.
    """
    batch_size = 16
    input_dim = 100
    hidden_dim = 50
    latent_dim = 32
    output_dim = 80
    batchnorm_momentum = 0.8

    # Create a random input tensor.
    x = torch.randn(batch_size, input_dim)

    # Create an encoder and decoder.
    encoder = Encoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        batchnorm_momentum=batchnorm_momentum,
    )

    decoder = Decoder(
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        batchnorm_momentum=batchnorm_momentum,
    )

    # Test forward pass through encoder and decoder.
    embeddings = encoder(x)
    output = decoder(embeddings)

    # Check output shape.
    assert output.shape == (batch_size, output_dim)
