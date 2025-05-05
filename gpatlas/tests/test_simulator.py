"""
Tests for the genotype-phenotype simulator.

These tests verify statistical properties of simulated datasets, including:
- Minor allele frequency distribution
- Heritability
- Pleiotropy and epistasis rates
- Environmental effects
"""

from collections import defaultdict

import numpy as np
import pytest
from sklearn.linear_model import Lasso

from gpatlas.simulators.simulator import (
    GenotypePhenotypeSimulator,
    SimulationData,
    SimulationParameters,
)


@pytest.mark.parametrize("minor_allele_freq_range", [(0.0, 0.5), (0.01, 0.05), (0.4, 0.6)])
def test_maf_distribution(minor_allele_freq_range):
    """Test that the minor allele frequency distribution matches the specified range."""
    params = SimulationParameters(
        seed=42,
        num_samples=500,
        num_loci=1000,
        minor_allele_freq_range=minor_allele_freq_range,
    )

    simulator = GenotypePhenotypeSimulator(params)
    data = simulator.simulate()

    assert data.allele_frequencies is not None

    assert np.min(data.allele_frequencies) >= params.minor_allele_freq_range[0]
    assert np.max(data.allele_frequencies) <= params.minor_allele_freq_range[1]

    # Check MAF distribution uniformity using Kolmogorov-Smirnov test
    scaled_mafs = (data.allele_frequencies - params.minor_allele_freq_range[0]) / (
        params.minor_allele_freq_range[1] - params.minor_allele_freq_range[0]
    )

    # Compare empirical CDF with uniform CDF.
    sorted_scaled_mafs = np.sort(scaled_mafs)
    n = len(sorted_scaled_mafs)
    uniform_cdf = np.arange(1, n + 1) / n
    max_diff = np.max(np.abs(sorted_scaled_mafs - uniform_cdf))

    # Critical value for KS test at alpha=0.05.
    critical_value = 1.36 / np.sqrt(n)
    assert max_diff < critical_value


def test_genotype_distribution():
    """
    Test that the genotype distribution matches expected frequencies for both diploid and haploid.
    """
    # Test diploid genotypes
    params = SimulationParameters(
        seed=42,
        num_samples=1000,
        num_loci=100,
        num_phenotypes=3,
        genome_type="diploid",
    )

    simulator = GenotypePhenotypeSimulator(params)
    data = simulator.simulate()

    assert data.genotypes is not None

    # Check range of values (should be 0, 1, or 2 for diploid)
    unique_values = np.unique(data.genotypes)
    assert set(unique_values).issubset({0, 1, 2})

    # Test haploid genotypes
    params = params.model_copy(update={"genome_type": "haploid"})

    simulator = GenotypePhenotypeSimulator(params)
    data = simulator.simulate()

    assert data.genotypes is not None

    # Check range of values (should be 0 or 1 for haploid)
    unique_values = np.unique(data.genotypes)
    assert set(unique_values).issubset({0, 1})


@pytest.mark.parametrize("effect_size_distribution", ["normal", "exponential", "gamma"])
def test_effect_size_distributions(effect_size_distribution):
    """
    Test that effect sizes match the specified distribution.

    TODO: check the mean and variance of the effect sizes for all distributions.
    """
    params = SimulationParameters(
        seed=42,
        num_samples=1000,
        num_loci=1000,
        num_phenotypes=10,
        num_causal_loci_per_phenotype=30,
        effect_size_distribution=effect_size_distribution,
        effect_size_std=1.0,
    )

    simulator = GenotypePhenotypeSimulator(params)
    data = simulator.simulate()

    all_effects = []
    for _, effects in data.effect_sizes.items():
        all_effects.extend(effects.values())

    assert len(all_effects)

    if effect_size_distribution == "normal":
        assert np.isclose(np.mean(all_effects), 0.0, atol=0.1)
        assert np.isclose(np.std(all_effects), 1.0, atol=0.1)

    # For exponential and gamma, check both positive and negative values.
    if effect_size_distribution in ["exponential", "gamma"]:
        pos_count = sum(1 for e in all_effects if e > 0)
        neg_count = sum(1 for e in all_effects if e < 0)

        assert pos_count > 0
        assert neg_count > 0

        # Roughly equal numbers of positive and negative values.
        ratio = pos_count / (pos_count + neg_count)
        assert np.isclose(ratio, 0.5, atol=0.1)


def test_heritability_sibling_correlation():
    """
    Test heritability using a sibling correlation approach.

    Siblings share the same genetic factors but have different environmental factors.
    Their correlation should increase with higher heritability values.
    """
    heritabilities = [0.0, 0.3, 0.7, 1.0]
    correlations_by_phenotype = defaultdict(list)

    params = SimulationParameters(
        seed=42,
        num_samples=1000,
        num_loci=200,
        num_phenotypes=5,
        num_causal_loci_per_phenotype=20,
        # Disable pleiotropy and epistasis for simpler analysis.
        pleiotropy_probability=0.0,
        epistasis_probability=0.0,
    )

    # For each heritability value, calculate sibling correlations.
    for heritability in heritabilities:
        params.broad_sense_heritability = heritability
        simulator = GenotypePhenotypeSimulator(params)
        sibling_1_data = simulator.simulate()

        assert sibling_1_data.genotypes is not None
        assert sibling_1_data.phenotypes is not None

        # For each phenotype, calculate sibling correlation.
        for phenotype_idx in range(params.num_phenotypes):
            sibling_1_phenotype = sibling_1_data.phenotypes[:, phenotype_idx]

            # Skip phenotypes with near-zero variance.
            if np.var(sibling_1_phenotype) < 1e-6:
                continue

            # Create "siblings" by copying genotypes and regenerating environment.
            sibling_2_data = SimulationData(params=params)
            sibling_2_data.genotypes = sibling_1_data.genotypes.copy()
            sibling_2_data.effect_sizes = sibling_1_data.effect_sizes.copy()
            sibling_2_data.causal_loci = sibling_1_data.causal_loci.copy()
            sibling_2_data.epistatic_effects = sibling_1_data.epistatic_effects.copy()

            # Generate new environmental factors and effects.
            sibling_2_data.environmental_factors = simulator._generate_environmental_factors()
            sibling_2_data.environmental_effects = simulator._generate_environmental_effects()
            sibling_2_data.environmental_noise = simulator._generate_environmental_noise()

            # Calculate phenotypes for siblings with same genetics but different environment.
            sibling_2_data.phenotypes = GenotypePhenotypeSimulator._calculate_phenotypes(
                params, sibling_2_data
            )

            # Calculate correlation between siblings.
            sibling_correlation = np.corrcoef(
                sibling_1_phenotype, sibling_2_data.phenotypes[:, phenotype_idx]
            )[0, 1]

            correlations_by_phenotype[phenotype_idx].append(sibling_correlation)

            # For zero heritability, siblings should have low correlation.
            if heritability == 0:
                assert np.isclose(sibling_correlation, 0.0, atol=0.1)

            # For perfect heritability, siblings should have high correlation.
            if heritability == 1.0:
                assert np.isclose(sibling_correlation, 1.0, atol=0.1)

    # For each phenotype, check that correlations increase with heritability.
    for phenotype_idx, correlations in correlations_by_phenotype.items():
        if len(correlations) == len(heritabilities):
            for i in range(1, len(correlations)):
                assert correlations[i] > correlations[i - 1], (
                    f"Sibling correlation should increase with heritability "
                    f"for phenotype {phenotype_idx}, "
                    f"but got {correlations[i-1]} for h2={heritabilities[i-1]} and "
                    f"{correlations[i]} for h2={heritabilities[i]}"
                )


@pytest.mark.parametrize("env_factors_weight", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("heritability", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
def test_phenotype_variance(heritability, env_factors_weight):
    """
    Test that the phenotypic variance is close to 1.0 for the given parameters.
    """
    params = SimulationParameters(
        seed=40,
        num_samples=10000,
        num_loci=100,
        num_phenotypes=3,
        num_causal_loci_per_phenotype=10,
        pleiotropy_probability=0.0,
        epistasis_probability=0.0,
        env_factors_weight=env_factors_weight,
        broad_sense_heritability=heritability,
    )

    simulator = GenotypePhenotypeSimulator(params)
    data = simulator.simulate()

    assert data.phenotypes is not None
    phenotype_variances = np.var(data.phenotypes, axis=0)

    # The tolerance was chosen empirically.
    assert np.allclose(phenotype_variances, 1.0, atol=0.1)


@pytest.mark.parametrize("pleiotropy_probability", [0.0, 0.3, 0.6, 1.0])
def test_pleiotropy_rate(pleiotropy_probability):
    """Test that the pleiotropy rate matches the specified probability."""
    params = SimulationParameters(
        seed=42,
        num_samples=500,
        num_loci=1000,
        num_phenotypes=10,
        num_causal_loci_per_phenotype=30,
        pleiotropy_probability=pleiotropy_probability,
    )

    simulator = GenotypePhenotypeSimulator(params)
    data = simulator.simulate()

    # Count pleiotropic loci.
    locus_phenotype_count = defaultdict(int)
    for _, loci in data.causal_loci.items():
        for locus in loci:
            locus_phenotype_count[locus] += 1

    pleiotropic_loci = [locus for locus, count in locus_phenotype_count.items() if count > 1]

    num_causal_loci = len(locus_phenotype_count)
    num_pleiotropic_loci = len(pleiotropic_loci)

    assert len(data.pleiotropic_effects) == num_pleiotropic_loci
    assert np.isclose(
        num_pleiotropic_loci / num_causal_loci, params.pleiotropy_probability, atol=0.05
    )


@pytest.mark.parametrize("epistasis_probability", [0.0, 0.3, 0.6, 1.0])
def test_epistasis_rate(epistasis_probability):
    """Test that the epistasis rate matches the specified probability."""
    params = SimulationParameters(
        seed=42,
        num_samples=5000,
        num_loci=500,
        num_phenotypes=20,
        num_causal_loci_per_phenotype=10,
        epistasis_probability=epistasis_probability,
    )

    simulator = GenotypePhenotypeSimulator(params)
    data = simulator.simulate()

    # Count potential and actual epistatic interactions.
    num_potential_interactions = 0
    for _, loci in data.causal_loci.items():
        n_loci = len(loci)
        num_potential_interactions += (n_loci * (n_loci - 1)) // 2

    epistatic_interactions_count = len(data.epistatic_effects)
    epistasis_rate = epistatic_interactions_count / num_potential_interactions
    assert np.isclose(epistasis_rate, params.epistasis_probability, atol=0.05)


@pytest.mark.parametrize("genome_type", ["haploid", "diploid"])
@pytest.mark.parametrize("heritability", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
def test_heritability_with_model_fit(genome_type, heritability):
    """
    Check that the heritability is close to the target value when fitting a linear model
    to simulated data with only additive effects.
    """
    params = SimulationParameters(
        seed=42,
        genome_type=genome_type,
        num_samples=1000,
        num_loci=100,
        num_phenotypes=3,
        num_causal_loci_per_phenotype=10,
        pleiotropy_probability=0.0,
        epistasis_probability=0.0,
        broad_sense_heritability=heritability,
    )

    simulator = GenotypePhenotypeSimulator(params)
    data = simulator.simulate()

    assert data.genotypes is not None
    assert data.phenotypes is not None

    # Fit a linear model to the data.
    # The value of `alpha` was chosen empirically to ensure the model is not too strict.
    model = Lasso(alpha=0.03)
    model.fit(data.genotypes, data.phenotypes)

    # Check that the coefficient of determination is close to the target heritability.
    assert np.isclose(model.score(data.genotypes, data.phenotypes), heritability, atol=0.1)


def test_save_functionality(tmp_path):
    """Test that the save functionality works as expected."""
    params = SimulationParameters(
        seed=42,
        num_samples=100,
        num_loci=50,
        num_phenotypes=3,
    )
    simulator = GenotypePhenotypeSimulator(params)
    data = simulator.simulate()
    data.save(tmp_path)

    assert (tmp_path / "train-data.h5").exists()
    assert (tmp_path / "test-data.h5").exists()
    assert (tmp_path / "params.yaml").exists()
