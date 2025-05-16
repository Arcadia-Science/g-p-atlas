"""
This module implements a rudimentary simulator for genotype-phenotype datasets.

It supports the following features:
- Additive effects.
- Gene-gene interactions (epistasis).
- Pleiotropy (genes affecting multiple phenotypes).
- Environmental effects and noise.

It does not support:
- Gene-environment interactions.
- Linkage disequilibrium.
- Dominance or other non-additive effects.

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
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

EPS = 1e-6


class SimulationParameters(BaseModel):
    """Parameters defining a genotype-phenotype simulation."""

    model_config = ConfigDict()

    seed: int = 42

    # The genome type determines the count of minor alleles at each locus.
    # (0 or 1 for haploid, and 0, 1, or 2 for diploid.)
    genome_type: Literal["diploid", "haploid"] = "haploid"

    # The number of samples (or individuals, or animals, etc.) to simulate.
    num_samples: int = 1000

    # The number of segregating loci to simulate.
    num_loci: int = 100

    # The range of minor allele frequencies to use when simulating loci.
    # To specify a single MAF, use the same value for both elements of the tuple.
    minor_allele_frequency_range: tuple[float, float] = (0.5, 0.5)

    # The number of phenotypes to simulate.
    num_phenotypes: int = 10

    # The number of additive causal loci per phenotype.
    num_causal_loci_per_phenotype: int = 5

    # The probability that a causal locus is pleiotropic (that is, has an effect on two phenotypes).
    # (This is not the same as the probability that a given pair of phenotypes
    # shares a causal locus.)
    pleiotropy_probability: float = 0.2

    # The probability that a pair of causal loci for a given phenotype are epistatic
    # (that is, have an interaction effect on the phenotype).
    epistasis_probability: float = 0.2

    # Proportion of phenotypic variance explained by genetic effects.
    # If a list, there must be one value per phenotype.
    broad_sense_heritability: float | list[float] = 0.6

    # Proportion of phenotypic variance explained by narrow-sense heritability
    # (additive genetic effects).
    # If a list, there must be one value per phenotype.
    narrow_sense_heritability: float | list[float] = 0.4

    # The type and scale of the effect size distribution.
    # TODO: add separate parameters for the distributions to use for additive, epistatic,
    # and environmental effects.
    effect_size_distribution: Literal["normal", "exponential", "gamma"] = "normal"
    effect_size_scale: float = 1.0

    # Proportion of environmental variance allocated to environmental factors vs. random noise.
    # To disable environmental effects (leaving only random noise), set `env_factors_weight` to 0.0.
    env_factors_weight: float = 0.5

    # The number of environmental factors (there must be at least one,
    # even if `env_factors_weight` is 0.0).
    num_environmental_factors: int = 1

    # The probability that a given environmental factor has an effect on a given phenotype.
    env_influence_probability: float = 0.2

    @field_validator("broad_sense_heritability", "narrow_sense_heritability")
    @classmethod
    def validate_heritability_range(cls, value: float | list[float]) -> float | list[float]:
        """
        Validate that heritability values are between 0 and 1.
        """
        if not isinstance(value, list):
            heritabilities = [value]
        else:
            heritabilities = value

        for h in heritabilities:
            if h < 0 or h > 1:
                raise ValueError(f"Heritability value {h} must be between 0 and 1")
        return value

    @model_validator(mode="after")
    def validate_heritability_hierarchy(cls, data):
        """
        Validate that broad sense heritability is greater than or equal
        to narrow sense heritability.
        """
        broad = data.broad_sense_heritability
        narrow = data.narrow_sense_heritability

        if isinstance(broad, list) and isinstance(narrow, list):
            if len(broad) != len(narrow):
                raise ValueError("If both heritabilities are lists, they must have the same length")
            for b, n in zip(broad, narrow, strict=True):
                if b < n:
                    raise ValueError(
                        f"Broad sense heritability ({b}) must be greater than or equal "
                        f"to narrow sense heritability ({n})"
                    )
        elif isinstance(broad, list) or isinstance(narrow, list):
            raise ValueError(
                "Either both heritabilities must be lists, or both must be single values"
            )
        else:
            if broad < narrow:
                raise ValueError(
                    f"Broad sense heritability ({broad}) must be greater than "
                    f"or equal to narrow sense heritability ({narrow})"
                )
        return data


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

    # A dict of dicts of additive effect sizes for each phenotype and locus.
    # Shape: `{phenotype_ind: {locus_ind: effect_size}}`
    additive_effect_sizes: dict[int, dict[int, float]] = {}

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
                "additive_effect_sizes",
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
            additive_effect_sizes=effect_sizes,
        )

        data.pleiotropic_effects = self._generate_pleiotropic_effects(
            data.causal_loci, data.additive_effect_sizes
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
            low=params.minor_allele_frequency_range[0],
            high=params.minor_allele_frequency_range[1],
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

        TODO: use a more realistic distribution to generate the pleiotropic effects
        that allows for covariance between the effect sizes.

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

    def _scale_additive_genetic_effects(
        self, data: SimulationData, phenotype_ind: int, scale_factor: float
    ) -> None:
        """
        Scale the additive genetic effect sizes for a given phenotype.
        """
        for locus_ind in data.additive_effect_sizes[phenotype_ind]:
            data.additive_effect_sizes[phenotype_ind][locus_ind] *= scale_factor

        for ind, (phenotype_1_ind, phenotype_2_ind, locus_ind, effect_size) in enumerate(
            data.pleiotropic_effects
        ):
            if phenotype_2_ind == phenotype_ind:
                data.pleiotropic_effects[ind] = (
                    phenotype_1_ind,
                    phenotype_2_ind,
                    locus_ind,
                    effect_size * scale_factor,
                )

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

    def _scale_epistatic_genetic_effects(
        self, data: SimulationData, phenotype_ind: int, scale_factor: float
    ) -> None:
        """
        Scale the epistatic genetic effect sizes for a given phenotype.
        """
        for ind, (phenotype_1_ind, locus_1_ind, locus_2_ind, effect_size) in enumerate(
            data.epistatic_effects
        ):
            if phenotype_1_ind == phenotype_ind:
                data.epistatic_effects[ind] = (
                    phenotype_1_ind,
                    locus_1_ind,
                    locus_2_ind,
                    effect_size * scale_factor,
                )

    def _scale_environmental_effects(
        self, data: SimulationData, phenotype_ind: int, scale_factor: float
    ) -> None:
        """
        Scale the environmental effect sizes for a given phenotype.
        """
        for env_factor_ind in data.environmental_effects[phenotype_ind]:
            data.environmental_effects[phenotype_ind][env_factor_ind] *= scale_factor

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

    def _calculate_phenotypes(
        self, params: SimulationParameters, data: SimulationData
    ) -> NDArray[np.float64]:
        """
        Calculate phenotypes by combining all genetic and environmental effects with the appropriate
        scale factors to achieve the target narrow- and broad-sense heritabilities
        for each phenotype.

        Note: the genetic effects are scaled in two steps.
        First, the additive and epistatic effects are scaled and combined,
        then the combined genetic effects are scaled to achieve the target broad-sense heritability
        when combined with the environmental effects.
        This two-step scaling is necessary because the additive and epistatic effects
        may not be mean-centered, so the actual variance of the combined genetic effects
        may differ from the desired broad-sense heritability without the second scaling step.
        """
        if (
            data.genotypes.size == 0
            or data.environmental_factors.size == 0
            or data.environmental_noise.size == 0
        ):
            raise ValueError("Required arrays must be populated before calculating phenotypes")

        # Create lists of heritabilities for convenience.
        broad_sense_heritabilities = params.broad_sense_heritability
        if not isinstance(broad_sense_heritabilities, list):
            broad_sense_heritabilities = [broad_sense_heritabilities] * params.num_phenotypes

        narrow_sense_heritabilities = params.narrow_sense_heritability
        if not isinstance(narrow_sense_heritabilities, list):
            narrow_sense_heritabilities = [narrow_sense_heritabilities] * params.num_phenotypes

        additive_genetic_effects = np.zeros((params.num_samples, params.num_phenotypes))
        epistatic_genetic_effects = np.zeros((params.num_samples, params.num_phenotypes))
        environmental_effects = np.zeros((params.num_samples, params.num_phenotypes))

        # Additive effects (including pleiotropic effects).
        for phenotype_ind in range(params.num_phenotypes):
            for locus_ind, effect_size in data.additive_effect_sizes[phenotype_ind].items():
                additive_genetic_effects[:, phenotype_ind] += (
                    data.genotypes[:, locus_ind] * effect_size
                )

        # Epistatic effects.
        for phenotype_ind in range(params.num_phenotypes):
            for other_phenotype_ind, locus_ind1, locus_ind2, effect_size in data.epistatic_effects:
                if other_phenotype_ind == phenotype_ind:
                    # We use a multiplicative model in which the interaction effect is
                    # proportional to the product of the allele counts.
                    interaction_effects = (
                        data.genotypes[:, locus_ind1] * data.genotypes[:, locus_ind2] * effect_size
                    )
                    epistatic_genetic_effects[:, phenotype_ind] += interaction_effects

        # Environmental effects.
        for phenotype_ind in range(params.num_phenotypes):
            for env_factor_ind, effect_size in data.environmental_effects[phenotype_ind].items():
                environmental_effects[:, phenotype_ind] += (
                    data.environmental_factors[:, env_factor_ind] * effect_size
                )

        # Construct the final phenotypes by scaling the genetic and environmental effects
        # to achieve the desired heritability for each phenotype.
        phenotypes = np.zeros((params.num_samples, params.num_phenotypes))
        for phenotype_ind in range(params.num_phenotypes):
            broad_sense_heritability = broad_sense_heritabilities[phenotype_ind]
            narrow_sense_heritability = narrow_sense_heritabilities[phenotype_ind]

            additive_genetic_var = np.var(additive_genetic_effects[:, phenotype_ind])
            epistatic_genetic_var = np.var(epistatic_genetic_effects[:, phenotype_ind])

            if epistatic_genetic_var < EPS and broad_sense_heritability > narrow_sense_heritability:
                print(
                    "Warning: the target broad-sense heritability is greater than "
                    "the target narrow-sense heritability, but the epistatic genetic "
                    "effects have no variance, so the actual narrow-sense heritability "
                    "will be equal to broad-sense heritability."
                )

            additive_genetic_scale_factor = np.sqrt(
                narrow_sense_heritability
                / (additive_genetic_var if additive_genetic_var > EPS else EPS)
            )

            epistatic_genetic_scale_factor = np.sqrt(
                (broad_sense_heritability - narrow_sense_heritability)
                / (epistatic_genetic_var if epistatic_genetic_var > EPS else EPS)
            )

            # Scale the additive and epistatic genetic effects to achieve the target
            # narrow-sense heritability relative to the target broad-sense heritability.
            all_genetic_effects = (
                additive_genetic_effects[:, phenotype_ind] * additive_genetic_scale_factor
                + epistatic_genetic_effects[:, phenotype_ind] * epistatic_genetic_scale_factor
            )

            # Scale the combined genetic effects to achieve the target broad-sense heritability.
            all_genetic_effects_var = np.var(all_genetic_effects)
            env_factors_var = np.var(environmental_effects[:, phenotype_ind])

            all_genetic_effects_scale_factor = np.sqrt(
                broad_sense_heritability
                / (all_genetic_effects_var if all_genetic_effects_var > EPS else EPS)
            )

            if all_genetic_effects_var < EPS and broad_sense_heritability > 0:
                raise ValueError(
                    "Genetic effects have no variance, but the target broad-sense heritability "
                    "is non-zero and cannot be achieved."
                )

            # Calculate total environmental variance needed.
            total_env_var_needed = 1 - broad_sense_heritability

            # Split the environmental variance between structured effects and random noise.
            env_effects_scale_factor: float
            env_noise_scale_factor: float
            if env_factors_var > EPS:
                env_effects_scale_factor = np.sqrt(
                    (total_env_var_needed * params.env_factors_weight) / env_factors_var
                )
                env_noise_scale_factor = np.sqrt(
                    total_env_var_needed * (1 - params.env_factors_weight)
                )
            else:
                # If there are no environmental factors, all environmental variance
                # comes from random noise.
                env_effects_scale_factor = 0
                env_noise_scale_factor = np.sqrt(total_env_var_needed)

            # Combine all effects.
            phenotypes[:, phenotype_ind] = (
                all_genetic_effects * all_genetic_effects_scale_factor
                + environmental_effects[:, phenotype_ind] * env_effects_scale_factor
                + data.environmental_noise[:, phenotype_ind] * env_noise_scale_factor
            )

            # Scale all of the effect sizes so that they match the scale of their corresponding
            # contributions to the phenotype.
            self._scale_additive_genetic_effects(
                data,
                phenotype_ind,
                additive_genetic_scale_factor * all_genetic_effects_scale_factor,
            )
            self._scale_epistatic_genetic_effects(
                data,
                phenotype_ind,
                epistatic_genetic_scale_factor * all_genetic_effects_scale_factor,
            )
            self._scale_environmental_effects(data, phenotype_ind, env_effects_scale_factor)

        # For debugging purposes, store the genetic and environmental effects.
        self.additive_genetic_effects = additive_genetic_effects
        self.epistatic_genetic_effects = epistatic_genetic_effects
        self.environmental_effects = environmental_effects

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
