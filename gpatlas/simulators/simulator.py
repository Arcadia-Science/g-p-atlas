#!/usr/bin/env python
"""
This module implements a simulator for genotype-phenotype relationships with support for:
- Additive effects
- Gene-gene interactions (epistasis)
- Pleiotropy (genes affecting multiple phenotypes)
- Environmental effects

"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import click
import h5py
import numpy as np
import yaml
from numpy.typing import NDArray
from pydantic import BaseModel


class SimulationParameters(BaseModel):
    """Parameters defining a genotype-phenotype simulation."""

    seed: int = 42

    num_samples: int = 1000
    num_loci: int = 1000
    minor_allele_freq_range: tuple[float, float] = (0.05, 0.5)
    genome_type: Literal["diploid", "haploid"] = "diploid"

    num_phenotypes: int = 30
    num_causal_loci_per_phenotype: int = 10

    # Proportion of variance explained by genetics.
    heritability: float | list[float] = 0.6

    pleiotropy_probability: float = 0.2
    epistasis_probability: float = 0.2

    # The parameters of the effect size distribution.
    effect_size_distribution: Literal["normal", "exponential", "gamma"] = "normal"
    effect_size_mean: float = 0.0
    effect_size_std: float = 1.0

    # Environmental factors. There must be at least one environmental factor.
    # (set env_factor_weight to 0 to disable environmental effects)
    env_factor_weight: float = 0.5
    num_environmental_factors: int = 1
    env_influence_probability: float = 0.2
    env_effect_size_range: tuple[float, float] = (0.0, 1.0)


@dataclass
class SimulationData:
    """Container for simulation input and output data."""

    params: SimulationParameters

    # Genotypes as an array of alternate/minor allele counts at each locus in each sample.
    # Shape: (num_samples, num_loci).
    genotypes: NDArray[np.int_] | None = field(default=None, repr=False)

    # Minor allele frequencies at each locus.
    # Shape: (num_loci,).
    allele_frequencies: NDArray[np.float64] | None = field(default=None, repr=False)

    # Phenotypes as an array of phenotype values for each sample.
    # Shape: (num_samples, num_phenotypes).
    phenotypes: NDArray[np.float64] | None = field(default=None, repr=False)

    # Environmental factors.
    # Shape: (num_samples, num_environmental_factors).
    environmental_factors: NDArray[np.float64] | None = field(default=None, repr=False)

    # A dict of lists of causal loci for each phenotype.
    causal_loci: dict[int, list[int]] = field(default_factory=dict)

    # A dict of dicts of effect sizes for each phenotype and locus.
    effect_sizes: dict[int, dict[int, float]] = field(default_factory=dict)

    # A list of tuples of pleiotropic effects.
    # Shape: (phenotype1, phenotype2, locus)
    pleiotropic_effects: list[tuple[int, int, float]] = field(default_factory=list)

    # A list of tuples of epistatic effects.
    # Shape: (phenotype, locus1, locus2, effect)
    epistatic_effects: list[tuple[int, int, int, float]] = field(default_factory=list)

    # A dict of dicts of environmental effects for each phenotype and environmental factor.
    env_effects: dict[int, dict[int, float]] = field(default_factory=dict)

    def split_train_test(self) -> tuple[SimulationData, SimulationData]:
        """
        Split the data into training and test sets.
        """
        raise NotImplementedError("Not implemented yet.")

    def save(self, dirpath: str | Path) -> None:
        """
        Save the simulated genotype and phenotype data as a pair of training and test datasets.

        TODO: handle the non-array data (e.g. causal loci, effect sizes, pleiotropic effects, etc).
        TODO: move the train/test split to a separate function.
        TODO: handle conversion to one-hot encoding properly (allele counts are lost).
        """
        dirpath = Path(dirpath)
        assert self.genotypes is not None
        assert self.phenotypes is not None

        # One-hot encode the genotypes.
        minor_allele_presence = self.genotypes > 0
        minor_allele_presence_one_hot = np.eye(2)[minor_allele_presence.astype(np.int_)]

        inds = np.arange(self.phenotypes.shape[0])
        np.random.shuffle(inds)
        num_training_samples = int(self.phenotypes.shape[0] * 0.8)
        train_inds = inds[:num_training_samples]
        test_inds = inds[num_training_samples:]

        for split, inds in [("train", train_inds), ("test", test_inds)]:
            with h5py.File(dirpath / f"{split}-data.h5", "w") as file:
                file.create_dataset("genotypes", data=minor_allele_presence_one_hot[inds])
                file.create_dataset("phenotypes", data=self.phenotypes[inds])


class GenotypePhenotypeSimulator:
    """Class for simulating genotype-phenotype relationships."""

    def __init__(self, params: SimulationParameters):
        """Initialize the simulator with parameters."""
        self.params = params
        np.random.seed(params.seed)

    def simulate(self) -> SimulationData:
        """Run a complete simulation."""
        data = SimulationData(params=self.params)

        data.genotypes, data.allele_frequencies = self._generate_genotypes()

        data.environmental_factors = self._generate_environmental_factors()

        data.causal_loci, data.effect_sizes = self._select_causal_loci()

        data.pleiotropic_effects = self._generate_pleiotropic_effects(
            data.causal_loci, data.effect_sizes
        )

        data.epistatic_effects = self._generate_epistatic_effects(data.causal_loci)

        data.env_effects = self._generate_environmental_effects()

        data.phenotypes = self._calculate_phenotypes(data)

        return data

    def _generate_effect_size(self) -> float:
        """
        Generate an effect size from the specified distribution.
        """
        params = self.params
        if params.effect_size_distribution == "normal":
            effect_size = np.random.normal(params.effect_size_mean, params.effect_size_std)
        elif params.effect_size_distribution == "exponential":
            effect_size = np.random.exponential(scale=params.effect_size_std)
        elif params.effect_size_distribution == "gamma":
            effect_size = np.random.gamma(shape=1.0, scale=params.effect_size_std)
        else:
            raise ValueError(f"Unknown effect size distribution: {params.effect_size_distribution}")

        # Randomly assign a sign.
        if params.effect_size_distribution in ["exponential", "gamma"]:
            effect_size *= np.random.choice([-1, 1])

        return effect_size

    def _generate_genotypes(self) -> tuple[NDArray[np.int_], NDArray[np.float64]]:
        """
        Generate genotypes for all samples.

        Returns:
            genotypes: NDArray[np.int_]
                An array of alternate/minor allele counts of shape (num_samples, num_loci).
            allele_freqs: NDArray[np.float64]
                An array of allele frequencies of shape (num_loci,) where each element is the
                frequency of the minor allele at a given locus.
        """
        params = self.params

        allele_freqs = np.random.uniform(
            low=params.minor_allele_freq_range[0],
            high=params.minor_allele_freq_range[1],
            size=params.num_loci,
        )

        genotypes = np.zeros((params.num_samples, params.num_loci), dtype=np.int_)
        for locus_ind in range(params.num_loci):
            genotypes[:, locus_ind] = np.random.binomial(
                n=(2 if params.genome_type == "diploid" else 1),
                p=allele_freqs[locus_ind],
                size=params.num_samples,
            )

        return genotypes, allele_freqs

    def _select_causal_loci(self) -> tuple[dict[int, list[int]], dict[int, dict[int, float]]]:
        """
        Select causal loci and assign effect sizes for each phenotype.

        Returns:
            causal_loci: dict[int, list[int]]
                A dictionary mapping each phenotype index to a list of causal loci indices.
            effect_sizes: dict[int, dict[int, float]]
                A dictionary mapping each phenotype index to a dictionary of effect sizes,
                whose keys are the locus indices and whose values are the effect sizes.
        """
        params = self.params
        causal_loci = {}
        effect_sizes = {}

        unselected_loci = set(np.arange(params.num_loci))
        for phenotype_ind in range(params.num_phenotypes):
            # Select causal loci without replacement.
            phenotype_causal_loci = np.random.choice(
                list(unselected_loci), size=params.num_causal_loci_per_phenotype, replace=False
            )

            # Remove the selected loci from the pool of all loci.
            unselected_loci -= set(phenotype_causal_loci)

            causal_loci[phenotype_ind] = list(phenotype_causal_loci)

            # Assign effect sizes based on specified distribution
            effects = [
                self._generate_effect_size() for _ in range(params.num_causal_loci_per_phenotype)
            ]
            locus_to_effect = {
                locus: effect for locus, effect in zip(phenotype_causal_loci, effects)
            }
            effect_sizes[phenotype_ind] = locus_to_effect

        return causal_loci, effect_sizes

    def _generate_pleiotropic_effects(
        self, causal_loci: dict[int, list[int]], effect_sizes: dict[int, dict[int, float]]
    ) -> list[tuple[int, int, float]]:
        """
        Generate pleiotropic effects between phenotypes.

        Returns:
            pleiotropic_effects: list[tuple[int, int, float]]
                A list of tuples of pleiotropic effects, where each tuple contains
                the indices of the two phenotypes and the locus that is pleiotropic.
        """
        pleiotropic_effects: list[tuple[int, int, float]] = []
        params = self.params

        for phenotype_ind in range(params.num_phenotypes):
            for other_phenotype_ind in range(phenotype_ind + 1, params.num_phenotypes):
                for locus_ind in causal_loci[phenotype_ind]:
                    if np.random.random() < params.pleiotropy_probability:
                        if locus_ind not in causal_loci[other_phenotype_ind]:
                            causal_loci[other_phenotype_ind].append(locus_ind)
                            effect_sizes[other_phenotype_ind][locus_ind] = (
                                self._generate_effect_size()
                            )
                            pleiotropic_effects.append(
                                (phenotype_ind, other_phenotype_ind, locus_ind)
                            )

        return pleiotropic_effects

    def _generate_epistatic_effects(
        self, causal_loci: dict[int, list[int]]
    ) -> list[tuple[int, int, int, float]]:
        """
        Generate epistatic (gene-gene interaction) effects.

        Returns:
            epistatic_effects: list[tuple[int, int, int, float]]
                A list of tuples of epistatic effects, where each tuple contains
                the indices of the phenotype, the two loci, and the effect size.
        """
        params = self.params
        epistatic_effects: list[tuple[int, int, int, float]] = []

        for phenotype_ind in range(params.num_phenotypes):
            causal_loci_inds = causal_loci[phenotype_ind]
            for ind, locus_ind in enumerate(causal_loci_inds):
                for other_locus_ind in causal_loci_inds[ind + 1 :]:
                    if np.random.random() < params.epistasis_probability:
                        effect_size = self._generate_effect_size()
                        epistatic_effects.append(
                            (phenotype_ind, locus_ind, other_locus_ind, effect_size)
                        )

        return epistatic_effects

    def _generate_environmental_factors(self) -> NDArray[np.float64]:
        """Generate environmental factors for all samples."""

        env_factors = np.random.normal(
            0, 1, (self.params.num_samples, self.params.num_environmental_factors)
        )
        env_factors *= np.float64(self.params.env_factor_weight)
        return env_factors

    def _generate_environmental_effects(self) -> dict[int, dict[int, float]]:
        """Generate phenotype-specific environmental effects.

        Returns:
            env_effects: dict[int, dict[int, float]]
                A dictionary mapping each phenotype index to a dictionary of environmental effects,
                where the keys are the environmental factor indices and the values
                are the effect sizes.
        """
        params = self.params
        env_effects = {}

        for phenotype_ind in range(params.num_phenotypes):
            phenotype_env_effects = {}

            for env_factor_ind in range(params.num_environmental_factors):
                if np.random.random() < params.env_influence_probability:
                    effect_size = np.random.uniform(*params.env_effect_size_range)
                    phenotype_env_effects[env_factor_ind] = effect_size

            env_effects[phenotype_ind] = phenotype_env_effects

        return env_effects

    def _calculate_phenotypes(self, data: SimulationData) -> NDArray[np.float64]:
        """Calculate phenotypes based on genetic and environmental factors."""

        assert data.genotypes is not None
        assert data.environmental_factors is not None

        params = self.params
        genetic_components = np.zeros((params.num_samples, params.num_phenotypes))
        environmental_components = np.zeros((params.num_samples, params.num_phenotypes))

        # Additive effects (including pleiotropic effects).
        for phenotype_ind in range(params.num_phenotypes):
            for locus_ind, effect_size in data.effect_sizes[phenotype_ind].items():
                genetic_components[:, phenotype_ind] += data.genotypes[:, locus_ind] * effect_size

        # Epistatic effects.
        for phenotype_ind in range(params.num_phenotypes):
            for other_phenotype_ind, locus_ind1, locus_ind2, effect_size in data.epistatic_effects:
                if other_phenotype_ind == phenotype_ind:
                    interaction = (
                        data.genotypes[:, locus_ind1] * data.genotypes[:, locus_ind2] * effect_size
                    )
                    genetic_components[:, phenotype_ind] += interaction

        # Environmental effects.
        for phenotype_ind in range(params.num_phenotypes):
            for env_factor_ind, effect_size in data.env_effects[phenotype_ind].items():
                environmental_components[:, phenotype_ind] += (
                    data.environmental_factors[:, env_factor_ind] * effect_size
                )

        # Scale genetic and environmental components to achieve target heritability.
        phenotypes = np.zeros((params.num_samples, params.num_phenotypes))
        for phenotype_ind in range(params.num_phenotypes):
            if isinstance(params.heritability, list):
                h2 = params.heritability[phenotype_ind]
            else:
                h2 = params.heritability

            # Calculate genetic and environmental variances
            genetic_var = np.var(genetic_components[:, phenotype_ind])
            env_component_var = np.var(environmental_components[:, phenotype_ind])

            if genetic_var <= 0:
                # No genetic variance, just use environmental component
                phenotypes[:, phenotype_ind] = environmental_components[:, phenotype_ind]
                continue

            # Add random environmental noise to match desired heritability
            env_noise = np.random.normal(0, 1, params.num_samples)
            
            # Calculate scalars to achieve target heritability
            if h2 > 0 and h2 < 1:
                # Scale genetic component to have variance h2
                genetic_scalar = np.sqrt(h2 / genetic_var)
                
                # Calculate total environmental variance needed for (1-h2)
                total_env_var_needed = (genetic_var * h2) * (1 - h2) / h2
                
                # Split between structured environmental effects and random noise
                if env_component_var > 0:
                    # Allocate half of environmental variance to structured effects and half to noise
                    env_component_scalar = np.sqrt((total_env_var_needed/2) / env_component_var)
                    noise_var = total_env_var_needed/2
                else:
                    # All environmental variance comes from noise
                    env_component_scalar = 0
                    noise_var = total_env_var_needed
                
                noise_scalar = np.sqrt(noise_var)
            else:
                # Edge cases (h2=0 or h2=1)
                if h2 == 0:
                    genetic_scalar = 0
                    env_component_scalar = 1
                    noise_scalar = 1
                else:  # h2 == 1
                    genetic_scalar = 1
                    env_component_scalar = 0
                    noise_scalar = 0
            
            # Combine components with proper scaling
            phenotypes[:, phenotype_ind] = (
                genetic_components[:, phenotype_ind] * genetic_scalar +
                environmental_components[:, phenotype_ind] * env_component_scalar +
                env_noise * noise_scalar
            )

        return phenotypes


@click.command()
@click.option("--output-dirpath", type=click.Path(path_type=Path), required=True)
@click.option("--config-filepath", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--split-train-test", type=bool, default=False)
def run_simulation(output_dirpath: Path, config_filepath: Path, split_train_test: bool):
    """
    Run a simulation and save the results to a file.
    """
    if output_dirpath.exists():
        click.confirm(
            f"Output directory {output_dirpath} already exists. Do you want to overwrite it?",
            abort=True,
        )

    output_dirpath.mkdir(parents=True, exist_ok=True)

    with open(config_filepath) as file:
        params = SimulationParameters(**yaml.safe_load(file))

    simulator = GenotypePhenotypeSimulator(params)
    data = simulator.simulate()

    data.save(output_dirpath)

    with open(output_dirpath / "params.yaml", "w") as file:
        yaml.safe_dump(params.model_dump(), file)

    click.echo(f"Simulation results saved to {output_dirpath}")


if __name__ == "__main__":
    run_simulation()
