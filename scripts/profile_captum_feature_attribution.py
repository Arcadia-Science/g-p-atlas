"""
This script was used to troubleshoot the performance of the `FeatureAblation` class
from the `captum` library.
"""

from timeit import timeit

import click
import torch
import torch.nn.functional as F
from captum.attr import FeatureAblation

from gpatlas.models import GPAtlasModel, GPAtlasModelConfig


@click.command()
@click.option("--num-loci", type=int, default=1000)
@click.option("--batch-size", type=int, default=10)
def main(num_loci: int, batch_size: int) -> None:
    """
    Profile the time it takes to call the forward function and the time it takes to call
    captum's `FeatureAblation.attribute` method.

    The `attribute` method should call the forward function once for each feature,
    so we should expect the time to call `attribute` to be `num_loci` times greater
    than the time to call `forward_func` directly.
    """
    num_phenotypes = 50
    num_alleles = 2

    # Create a GPA model with realistic-ish parameters.
    config = GPAtlasModelConfig(
        num_loci=num_loci,
        num_alleles=num_alleles,
        latent_dim=32,
        num_input_phenotypes=num_phenotypes,
        num_output_phenotypes=num_phenotypes,
        genotype_encoder_hidden_dim=128,
        phenotype_encoder_hidden_dim=32,
        genotype_decoder_hidden_dim=None,
        phenotype_decoder_hidden_dim=32,
    )

    model = GPAtlasModel(config=config)
    _ = model.eval()

    # Define the forward function for computing attributions for loci.
    def forward_func(genotypes: torch.Tensor, true_phenotypes: torch.Tensor) -> torch.Tensor:
        predicted_phenotypes = model.phenotype_decoder(model.genotype_encoder(genotypes))
        return F.mse_loss(predicted_phenotypes, true_phenotypes)

    genotypes = torch.randint(0, 2, (batch_size, num_loci * num_alleles), dtype=torch.float32)
    phenotypes = torch.rand((batch_size, num_phenotypes), dtype=torch.float32)

    # Call the `attribute` method once.
    ablator = FeatureAblation(forward_func)
    ablator_time = timeit(
        lambda: ablator.attribute(genotypes, additional_forward_args=phenotypes),
        number=1,
    )
    print(f"Time to call FeatureAblation.attribute once: {ablator_time:.2f} seconds")

    # Call the forward function directly once for each input feature.
    # This should be comparable to the time it takes to call `ablator.attribute` once.
    forward_func_time = timeit(
        lambda: forward_func(genotypes, phenotypes),
        number=(genotypes.shape[1]),
    )
    print(f"Time to call forward_func {num_loci} times: {forward_func_time:.2f} seconds")


if __name__ == "__main__":
    main()
