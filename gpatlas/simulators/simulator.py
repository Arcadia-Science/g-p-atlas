#!/usr/bin/env python
"""
This module implements a simulator for genotype-phenotype relationships with support for:
- Additive effects
- Gene-gene interactions (epistasis)
- Pleiotropy (genes affecting multiple phenotypes)
- Environmental effects

"""

from __future__ import annotations
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Literal

import click
import h5py
import numpy as np
import yaml
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

EPS = 1e-6


class SimulationParameters(BaseModel):
    """Parameters defining a genotype-phenotype simulation."""

    model_config = ConfigDict()

    seed: int = 42

    num_samples: int = 1000
    num_loci: int = 1000
    minor_allele_freq_range: tuple[float, float] = (0.05, 0.5)
    genome_type: Literal["diploid", "haploid"] = "diploid"

    num_phenotypes: int = 30
    num_causal_loci_per_phenotype: int = 10

    # The probability that a causal locus is pleiotropic (that is, has an effect on two phenotypes).
    # (This is not the same as the probability that a given pair of phenotypes
    # shares a causal locus.)
    pleiotropy_probability: float = 0.2

    # The probability that a pair of causal loci for a given phenotype are epistatic
    # (that is, have an interaction effect on the phenotype).
    epistasis_probability: float = 0.2

    # Proportion of variance explained by genetic effects.
    # To disable all environmental effects and environmental noise, set to 1.0.
    broad_sense_heritability: float | list[float] = 0.6

    # The parameters of the effect size distribution.
    effect_size_distribution: Literal["normal", "exponential", "gamma"] = "normal"
    effect_size_scale: float = 1.0

    # Environmental factors. There must be at least one environmental factor.
    # To disable environmental effects (leaving only random noise), set `env_factors_weight` to 0.0.
    num_environmental_factors: int = 1
    env_influence_probability: float = 0.2

    # Proportion of environmental variance allocated to environmental factors vs. random noise.
    # (0.0 = all random noise, 1.0 = all environmental factors).
    env_factors_weight: float = 0.5


class SimulationData(BaseModel):
    """Container for simulation input and output data."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    params: SimulationParameters

    # Genotypes as an array of alternate/minor allele counts at each locus in each sample.
    # Shape: `(num_samples, num_loci)`
    genotypes: NDArray[np.int_] = np.array([], dtype=np.int_)

    # Minor allele frequencies at each locus.
    # Shape: `(num_loci,)`
    allele_frequencies: NDArray[np.float64] = np.array([], dtype=np.float64)

    # Phenotypes as an array of phenotype values for each sample.
    # Shape: `(num_samples, num_phenotypes)`
    phenotypes: NDArray[np.float64] = np.array([], dtype=np.float64)

    # Environmental factors.
    # Shape: `(num_samples, num_environmental_factors)`
    environmental_factors: NDArray[np.float64] = np.array([], dtype=np.float64)

    # A dict of lists of causal loci for each phenotype.
    # Shape: `{phenotype_ind: [locus_ind]}`
    causal_loci: dict[int, list[int]] = {}

    # A dict of dicts of effect sizes for each phenotype and locus.
    # Shape: `{phenotype_ind: {locus_ind: effect_size}}`
    effect_sizes: dict[int, dict[int, float]] = {}

    # A list of tuples of pleiotropic effects.
    # Shape: `[(phenotype1, phenotype2, locus, effect_size)]`
    pleiotropic_effects: list[tuple[int, int, int, float]] = []

    # A list of tuples of epistatic effects.
    # Shape: `[(phenotype, locus1, locus2, effect_size)]`
    epistatic_effects: list[tuple[int, int, int, float]] = []

    # A dict of dicts of environmental effects for each phenotype and environmental factor.
    # Shape: `{phenotype_ind: {env_factor_ind: effect_size}}`
    environmental_effects: dict[int, dict[int, float]] = {}

    # Random environmental noise for each sample and phenotype.
    # Shape: `(num_samples, num_phenotypes)`
    environmental_noise: NDArray[np.float64] = np.array([], dtype=np.float64)

    def split_into_train_test(
        self, train_test_split: float = 0.8, seed: int = 42
    ) -> tuple[SimulationData, SimulationData]:
        """
        Split the dataset into training and test sets.
        """
        inds = np.arange(self.phenotypes.shape[0])

        np.random.seed(seed)
        np.random.shuffle(inds)

        # Note: this does indeed properly copy the numpy arrays.
        train_data = self.model_copy(deep=True)
        test_data = self.model_copy(deep=True)

        num_training_samples = int(self.phenotypes.shape[0] * train_test_split)
        train_inds = inds[:num_training_samples]
        test_inds = inds[num_training_samples:]

        # Split all of the sample-dependent arrays using the train/test indices.
        for attr in ["genotypes", "phenotypes", "environmental_factors", "environmental_noise"]:
            setattr(train_data, attr, getattr(self, attr)[train_inds])
            setattr(test_data, attr, getattr(self, attr)[test_inds])

        return train_data, test_data

    def one_hot_encode_genotypes(self, genotypes: NDArray[np.int_]) -> NDArray[np.int_]:
        """
        One-hot encode the minor allele counts in the `genotypes` array.
        """
        num_alleles = 2 if self.params.genome_type == "diploid" else 1
        return np.eye(num_alleles + 1, dtype=np.int_)[genotypes.astype(np.int_)]

    def to_hdf5(self, filepath: str | Path, one_hot_encode_genotypes: bool = True) -> None:
        """
        Dump the simulation data to an HDF5 file.
        """
        if self.genotypes.size == 0 or self.phenotypes.size == 0:
            raise ValueError("Genotypes and phenotypes must be populated before saving")

        with h5py.File(filepath, "w") as file:
            arrays_group = file.create_group("arrays")
            metadata_group = file.create_group("metadata")

            numpy_array_attrs = [
                "genotypes",
                "allele_frequencies",
                "phenotypes",
                "environmental_factors",
                "environmental_noise",
            ]
            for attr_name in numpy_array_attrs:
                data = getattr(self, attr_name)
                if attr_name == "genotypes" and one_hot_encode_genotypes:
                    arrays_group.create_dataset(attr_name, data=self.one_hot_encode_genotypes(data))
                else:
                    arrays_group.create_dataset(attr_name, data=data)

            json_like_attrs = [
                "causal_loci",
                "effect_sizes",
                "environmental_effects",
                "pleiotropic_effects",
                "epistatic_effects",
            ]
            for attr_name in json_like_attrs:
                metadata_group.create_dataset(
                    attr_name, data=json.dumps(getattr(self, attr_name), default=lambda x: str(x))
                )

            metadata_group.create_dataset("params", data=json.dumps(self.params.model_dump()))

            file.attrs["created_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def save_train_test(
        self,
        dirpath: str | Path,
        one_hot_encode_genotypes: bool = True,
        train_test_split: float = 0.8,
        seed: int = 42,
    ) -> None:
        """
        Save the simulated genotype and phenotype data as a pair of training and test datasets.
        """
        dirpath = Path(dirpath)
        dirpath.mkdir(parents=True, exist_ok=True)

        train_data, test_data = self.split_into_train_test(
            train_test_split=train_test_split, seed=seed
        )

        for name, dataset in [("train", train_data), ("test", test_data)]:
            dataset.to_hdf5(dirpath / f"{name}-data.h5", one_hot_encode_genotypes)

        with open(dirpath / "params.yaml", "w") as file:
            yaml.safe_dump(self.params.model_dump(), file)


class GenotypePhenotypeSimulator:
    """Class for simulating genotypes and phenotypes."""

    def __init__(self, params: SimulationParameters):
        """Initialize the simulator with parameters."""
        self.params = params
        np.random.seed(params.seed)

    def simulate(self) -> SimulationData:
        """Run a complete simulation."""
        genotypes, allele_frequencies = self._generate_genotypes()
        causal_loci, effect_sizes = self._select_causal_loci()

        data = SimulationData(
            params=self.params,
            genotypes=genotypes,
            allele_frequencies=allele_frequencies,
            causal_loci=causal_loci,
            effect_sizes=effect_sizes,
        )

        data.pleiotropic_effects = self._generate_pleiotropic_effects(
            data.causal_loci, data.effect_sizes
        )

        data.epistatic_effects = self._generate_epistatic_effects(data.causal_loci)

        data.environmental_factors = self._generate_environmental_factors()
        data.environmental_effects = self._generate_environmental_effects()
        data.environmental_noise = self._generate_environmental_noise()

        data.phenotypes = self._calculate_phenotypes(self.params, data)

        return data

    def _generate_effect_size(self) -> float:
        """
        Generate an effect size from the specified distribution.
        """
        params = self.params
        if params.effect_size_distribution == "normal":
            effect_size = np.random.normal(loc=0.0, scale=params.effect_size_scale)
        elif params.effect_size_distribution == "exponential":
            effect_size = np.random.exponential(scale=params.effect_size_scale)
        elif params.effect_size_distribution == "gamma":
            effect_size = np.random.gamma(shape=2.0, scale=params.effect_size_scale)
        else:
            raise ValueError(f"Unknown effect size distribution: {params.effect_size_distribution}")

        # Randomly assign a sign for distributions on [0, inf).
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
                A dictionary keyed by phenotype index, with values that are dictionaries
                of effect sizes, keyed by locus indices.
        """
        params = self.params
        causal_loci = {}
        effect_sizes = {}

        unselected_loci = set(np.arange(params.num_loci))
        for phenotype_ind in range(params.num_phenotypes):
            if len(unselected_loci) < params.num_causal_loci_per_phenotype:
                raise ValueError(
                    f"Not enough loci to select {params.num_causal_loci_per_phenotype} causal loci "
                    "for all phenotypes."
                )

            # Select causal loci without replacement.
            phenotype_causal_loci = np.random.choice(
                list(unselected_loci), size=params.num_causal_loci_per_phenotype, replace=False
            )

            # Remove the selected loci from the pool of all loci.
            unselected_loci -= set(phenotype_causal_loci)

            causal_loci[phenotype_ind] = list(phenotype_causal_loci)

            # Assign effect sizes based on specified distribution.
            phenotype_effect_sizes = [
                self._generate_effect_size() for _ in range(params.num_causal_loci_per_phenotype)
            ]
            effect_sizes[phenotype_ind] = {
                locus_ind: effect_size
                for locus_ind, effect_size in zip(
                    phenotype_causal_loci, phenotype_effect_sizes, strict=True
                )
            }

        return causal_loci, effect_sizes

    def _generate_pleiotropic_effects(
        self, causal_loci: dict[int, list[int]], effect_sizes: dict[int, dict[int, float]]
    ) -> list[tuple[int, int, int, float]]:
        """
        Generate pleiotropic effects between phenotypes by assigning a random subset
        of all causal loci to have an effect on a second phenotype.

        Updates the `causal_loci` and `effect_sizes` dictionaries in-place
        to include the pleiotropic effects.

        Returns:
            pleiotropic_effects: list[tuple[int, int, float]]
                A list of tuples of pleiotropic effects, where each tuple contains
                the indices of the two phenotypes and the locus that is pleiotropic.
        """
        params = self.params
        pleiotropic_effects: list[tuple[int, int, int, float]] = []

        causal_locus_to_phenotype = {
            locus_ind: phenotype_ind
            for phenotype_ind, locus_inds in causal_loci.items()
            for locus_ind in locus_inds
        }

        all_causal_locus_inds = list(causal_locus_to_phenotype.keys())
        all_phenotype_inds = set(causal_locus_to_phenotype.values())

        for locus_ind in all_causal_locus_inds:
            if np.random.random() < params.pleiotropy_probability:
                phenotype_ind = causal_locus_to_phenotype[locus_ind]
                second_phenotype_ind = np.random.choice(list(all_phenotype_inds - {phenotype_ind}))

                effect_size = self._generate_effect_size()
                causal_loci[second_phenotype_ind].append(locus_ind)
                effect_sizes[second_phenotype_ind][locus_ind] = effect_size
                pleiotropic_effects.append(
                    (phenotype_ind, second_phenotype_ind, locus_ind, effect_size)
                )

        return pleiotropic_effects

    def _generate_epistatic_effects(
        self, causal_loci: dict[int, list[int]]
    ) -> list[tuple[int, int, int, float]]:
        """
        Generate epistatic (gene-gene interaction) effects between causal loci.

        Returns:
            epistatic_effects: list[tuple[int, int, int, float]]
                A list of tuples of epistatic effects, where each tuple contains
                the indices of the phenotype, the two loci, and the effect size of the interaction.
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
        """
        Generate environmental factors for all samples.

        Enviromental factors represent the random environmental conditions to which
        each sample is exposed.

        The effect a given environmental condition has on a given phenotype is defined separately,
        in the `_generate_environmental_effects` method.

        Returns:
            env_factors: NDArray[np.float64]
                An array of environmental factors of shape (num_samples, num_environmental_factors).
        """

        env_factors = np.random.normal(
            0, 1, (self.params.num_samples, self.params.num_environmental_factors)
        )
        return env_factors

    def _generate_environmental_effects(self) -> dict[int, dict[int, float]]:
        """
        Generate phenotype-specific environmental effects.

        Returns:
            env_effects: dict[int, dict[int, float]]
                A dictionary mapping each phenotype index to a dictionary of environmental effects,
                where the keys are the environmental factor indices and the values
                are the effect sizes.
        """
        params = self.params
        env_effects = defaultdict(dict)

        for phenotype_ind in range(params.num_phenotypes):
            for env_factor_ind in range(params.num_environmental_factors):
                if np.random.random() < params.env_influence_probability:
                    effect_size = self._generate_effect_size()
                    env_effects[phenotype_ind][env_factor_ind] = effect_size

        return env_effects

    def _generate_environmental_noise(self) -> NDArray[np.float64]:
        """
        Generate random environmental noise for all samples.

        Returns:
            env_noise: NDArray[np.float64]
                An array of environmental noise of shape (num_samples, num_phenotypes).
        """
        return np.random.normal(0, 1, (self.params.num_samples, self.params.num_phenotypes))

    @staticmethod
    def _calculate_phenotypes(
        params: SimulationParameters, data: SimulationData
    ) -> NDArray[np.float64]:
        """
        Calculate phenotypes by combining genetic and environmental components.
        """
        if (
            data.genotypes.size == 0
            or data.environmental_factors.size == 0
            or data.environmental_noise.size == 0
        ):
            raise ValueError("Required arrays must be populated before calculating phenotypes")

        # The genetic and environmental components of the phenotypes.
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
                    # The interaction effect is proportional to the product of the allele counts.
                    interaction = (
                        data.genotypes[:, locus_ind1] * data.genotypes[:, locus_ind2] * effect_size
                    )
                    genetic_components[:, phenotype_ind] += interaction

        # Environmental effects.
        for phenotype_ind in range(params.num_phenotypes):
            for env_factor_ind, effect_size in data.environmental_effects[phenotype_ind].items():
                environmental_components[:, phenotype_ind] += (
                    data.environmental_factors[:, env_factor_ind] * effect_size
                )

        # Construct the final phenotypes by scaling the genetic and environmental components
        # to achieve the desired heritability for each phenotype.
        phenotypes = np.zeros((params.num_samples, params.num_phenotypes))
        heritability: float
        for phenotype_ind in range(params.num_phenotypes):
            if isinstance(params.broad_sense_heritability, list):
                heritability = params.broad_sense_heritability[phenotype_ind]
            else:
                heritability = params.broad_sense_heritability

            genetic_var = np.var(genetic_components[:, phenotype_ind])
            env_factors_var = np.var(environmental_components[:, phenotype_ind])

            # If there is no genetic variance, we set the heritability to 0
            # to avoid division by zero.
            if genetic_var <= EPS:
                heritability = 0

            genetic_scale_factor: float
            env_component_scale_factor: float
            env_noise_scale_factor: float

            if heritability < 0 or heritability > 1:
                raise ValueError(f"Invalid value for broad sense heritability: {heritability}")

            # Scale genetic component to have variance equal to `heritability`.
            genetic_scale_factor = np.sqrt(
                heritability / (genetic_var if genetic_var > EPS else 1.0)
            )

            # Calculate total environmental variance needed.
            total_env_var_needed = 1 - heritability

            # Split the environmental variance between structured effects and random noise.
            if env_factors_var > EPS:
                env_component_scale_factor = np.sqrt(
                    (total_env_var_needed * params.env_factors_weight) / env_factors_var
                )
                env_noise_scale_factor = np.sqrt(
                    total_env_var_needed * (1 - params.env_factors_weight)
                )
            else:
                # All environmental variance comes from random noise.
                env_component_scale_factor = 0
                env_noise_scale_factor = np.sqrt(total_env_var_needed)

            # Combine all components.
            phenotypes[:, phenotype_ind] = (
                genetic_components[:, phenotype_ind] * genetic_scale_factor
                + environmental_components[:, phenotype_ind] * env_component_scale_factor
                + data.environmental_noise[:, phenotype_ind] * env_noise_scale_factor
            )

        return phenotypes


@click.command()
@click.option("--output-dirpath", type=click.Path(path_type=Path), required=True)
@click.option("--config-filepath", type=click.Path(exists=True, path_type=Path), required=True)
def run_simulation(output_dirpath: Path, config_filepath: Path):
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
    data.save_train_test(output_dirpath)

    click.echo(f"Simulation results saved to {output_dirpath}")


if __name__ == "__main__":
    run_simulation()
