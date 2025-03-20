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

import numpy as np
from numpy.typing import NDArray


@dataclass
class SimulationParameters:
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

    def save(self, filepath: str | Path) -> None:
        """Save the simulation data to a file."""
        raise NotImplementedError("Saving is not implemented yet.")

    @classmethod
    def load(cls, filepath: str | Path) -> SimulationData:
        """Load simulation data from a file."""
        raise NotImplementedError("Loading is not implemented yet.")


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

        all_loci = set(np.arange(params.num_loci))
        for phenotype_ind in range(params.num_phenotypes):
            # Select causal loci without replacement.
            phenotype_causal_loci = np.random.choice(
                list(all_loci), size=params.num_causal_loci_per_phenotype, replace=False
            )

            # Remove the selected loci from the pool of all loci.
            all_loci -= set(phenotype_causal_loci)

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
        phenotypes = np.zeros((params.num_samples, params.num_phenotypes))

        # Additive effects (including pleiotropic effects).
        for phenotype_ind in range(params.num_phenotypes):
            for locus_ind, effect_size in data.effect_sizes[phenotype_ind].items():
                phenotypes[:, phenotype_ind] += data.genotypes[:, locus_ind] * effect_size

        # Epistatic effects.
        for phenotype_ind in range(params.num_phenotypes):
            for other_phenotype_ind, locus_ind1, locus_ind2, effect_size in data.epistatic_effects:
                if other_phenotype_ind == phenotype_ind:
                    interaction = (
                        data.genotypes[:, locus_ind1] * data.genotypes[:, locus_ind2] * effect_size
                    )
                    phenotypes[:, phenotype_ind] += interaction

        # Environmental effects.
        for phenotype_ind in range(params.num_phenotypes):
            for env_factor_ind, effect_size in data.env_effects[phenotype_ind].items():
                phenotypes[:, phenotype_ind] += (
                    data.environmental_factors[:, env_factor_ind] * effect_size
                )

        # Scale genetic component to achieve target heritability.
        for phenotype_ind in range(params.num_phenotypes):
            if isinstance(params.heritability, list):
                h2 = params.heritability[phenotype_ind]
            else:
                h2 = params.heritability

            genetic_var = np.var(phenotypes[:, phenotype_ind])

            if genetic_var > 0:
                # Calculate required environmental variance to achieve target heritability.
                env_var = genetic_var * (1 - h2) / h2

                # Add random environmental noise.
                env_noise = np.random.normal(0, np.sqrt(env_var), params.num_samples)
                phenotypes[:, phenotype_ind] += env_noise

        return phenotypes


def run_simulation_sweep(
    output_dir: str | Path,
    pleiotropy_values: list[float],
    epistasis_values: list[float],
    base_params: SimulationParameters | None = None,
) -> None:
    """
    Run a parameter sweep over pleiotropy and epistasis values.

    Args:
        output_dir: Directory to save simulation results
        pleiotropy_values: list of pleiotropy probabilities to test
        epistasis_values: list of epistasis probabilities to test
        base_params: Base parameters to use for all simulations
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if base_params is None:
        base_params = SimulationParameters()

    for pleio_prob in pleiotropy_values:
        for epist_prob in epistasis_values:
            params = SimulationParameters(
                **{
                    **base_params.__dict__,
                    "pleiotropy_probability": pleio_prob,
                    "epistasis_probability": epist_prob,
                }
            )

            simulator = GenotypePhenotypeSimulator(params)
            _ = simulator.simulate()
            # TODO: save the data here.

            print(f"Completed simulation with pleiotropy={pleio_prob}, epistasis={epist_prob}")


if __name__ == "__main__":
    # Example usage.
    params = SimulationParameters(
        num_samples=1000,
        num_loci=3000,
        num_phenotypes=30,
        num_causal_loci_per_phenotype=10,
        pleiotropy_probability=0.2,
        epistasis_probability=0.2,
        heritability=0.6,
        num_environmental_factors=2,
        env_influence_probability=0.2,
        env_effect_size_range=(0.0, 1.0),
        seed=42,
    )

    simulator = GenotypePhenotypeSimulator(params)
    data = simulator.simulate()

    assert data.genotypes is not None
    assert data.phenotypes is not None

    print(f"data.genotypes.shape: {data.genotypes.shape}")
    print(f"data.phenotypes.shape: {data.phenotypes.shape}")
