import json
from collections import defaultdict

import h5py
import numpy as np
import pytest
from sklearn.linear_model import Lasso

from gpatlas.simulators.simulator import (
    GenotypePhenotypeSimulator,
    SimulationData,
    SimulationParameters,
)

RANDOM_SEED = 42


@pytest.mark.parametrize("minor_allele_freq_range", [(0.0, 0.5), (0.01, 0.05), (0.4, 0.6)])
def test_maf_distribution(minor_allele_freq_range):
    """
    Test that the minor allele frequency distribution matches the specified range.
    """
    params = SimulationParameters(
        seed=RANDOM_SEED,
        num_samples=500,
        num_loci=1000,
        minor_allele_frequency_range=minor_allele_freq_range,
        broad_sense_heritability=0.0,
        narrow_sense_heritability=0.0,
    )

    simulator = GenotypePhenotypeSimulator(params)
    data = simulator.simulate()

    assert data.allele_frequencies is not None

    assert np.min(data.allele_frequencies) >= params.minor_allele_frequency_range[0]
    assert np.max(data.allele_frequencies) <= params.minor_allele_frequency_range[1]

    # Check MAF distribution uniformity using Kolmogorov-Smirnov test
    scaled_mafs = (data.allele_frequencies - params.minor_allele_frequency_range[0]) / (
        params.minor_allele_frequency_range[1] - params.minor_allele_frequency_range[0]
    )

    # Compare the empirical CDF with a uniform CDF.
    sorted_scaled_mafs = np.sort(scaled_mafs)
    n = len(sorted_scaled_mafs)
    uniform_cdf = np.arange(1, n + 1) / n
    max_diff = np.max(np.abs(sorted_scaled_mafs - uniform_cdf))

    # Critical value for KS test at alpha=0.05.
    critical_value = 1.36 / np.sqrt(n)
    assert max_diff < critical_value


def test_haploid_genotype_distribution():
    """
    Test that the genotype distribution matches expected frequencies for haploid genotypes.
    """
    num_loci = 100
    params = SimulationParameters(
        seed=RANDOM_SEED,
        num_samples=10000,
        num_loci=num_loci,
        num_phenotypes=3,
        genome_type="haploid",
        # Use a wide range of MAFs to test the distribution at both extremes.
        minor_allele_frequency_range=(0.1, 0.9),
    )

    simulator = GenotypePhenotypeSimulator(params)
    data = simulator.simulate()

    assert set(np.unique(data.genotypes)) == {0, 1}

    # Check that the count of minor alleles matches the expected frequencies.
    for locus_ind in range(num_loci):
        assert np.isclose(
            np.mean(data.genotypes[:, locus_ind] == 1),
            data.allele_frequencies[locus_ind],
            atol=0.05,
        )


def test_diploid_genotype_distribution():
    """
    Test that the genotype distribution matches expected frequencies for diploid genotypes.
    """
    num_loci = 100
    params = SimulationParameters(
        seed=RANDOM_SEED,
        num_samples=10000,
        num_loci=num_loci,
        num_phenotypes=3,
        genome_type="diploid",
        # Use a wide range of MAFs to test the distribution at both extremes.
        minor_allele_frequency_range=(0.1, 0.9),
    )

    simulator = GenotypePhenotypeSimulator(params)
    data = simulator.simulate()

    assert set(np.unique(data.genotypes)) == {0, 1, 2}

    # For each locus, check that the distribution follows Hardy-Weinberg Equilibrium
    # and that the mean matches the expected value (2 * minor allele frequency)
    for locus_ind in range(num_loci):
        minor_allele_freq = data.allele_frequencies[locus_ind]

        # Count occurrences of each genotype
        counts = np.bincount(data.genotypes[:, locus_ind], minlength=3)
        observed_frequencies = counts / params.num_samples

        # Expected frequencies for 0, 1, and 2 minor alleles, respectively.
        expected_frequencies = [
            (1 - minor_allele_freq) ** 2,
            2 * minor_allele_freq * (1 - minor_allele_freq),
            minor_allele_freq**2,
        ]

        # Check that observed frequencies are close to expected frequencies.
        assert np.allclose(observed_frequencies, expected_frequencies, atol=0.05)

        # Check that the mean matches twices the minor allele frequency.
        # (The expected mean is `0*P(0) + 1*P(1) + 2*P(2) = 2p`)
        expected_mean = 2 * minor_allele_freq
        observed_mean = np.mean(data.genotypes[:, locus_ind])
        assert np.isclose(observed_mean, expected_mean, atol=0.05)


@pytest.mark.parametrize("effect_size_distribution", ["normal", "exponential", "gamma"])
def test_effect_size_distributions(effect_size_distribution):
    """
    Test that effect sizes match the specified distribution.

    TODO: check the mean and variance of the effect sizes for all distributions.
    """
    params = SimulationParameters(
        seed=RANDOM_SEED,
        num_samples=1000,
        num_loci=1000,
        num_phenotypes=10,
        num_causal_loci_per_phenotype=30,
        effect_size_distribution=effect_size_distribution,
        effect_size_scale=1.0,
    )

    simulator = GenotypePhenotypeSimulator(params)
    data = simulator.simulate()

    all_effects = []
    for _, effects in data.additive_effect_sizes.items():
        all_effects.extend(effects.values())

    assert len(all_effects)

    if effect_size_distribution == "normal":
        # We cannot check the variance because the effect sizes are scaled.
        assert np.isclose(np.mean(all_effects), 0.0, atol=0.1)

    if effect_size_distribution in ["exponential", "gamma"]:
        assert np.mean(all_effects) > 0

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
        seed=RANDOM_SEED,
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
        # Only additive effects (all heritability is narrow-sense).
        params.narrow_sense_heritability = heritability
        params.broad_sense_heritability = heritability

        simulator = GenotypePhenotypeSimulator(params)
        sibling_1_data = simulator.simulate()

        # Create a second sibling with the same genetics but different environment.
        sibling_2_data = SimulationData(params=params)
        sibling_2_data.genotypes = sibling_1_data.genotypes.copy()
        sibling_2_data.additive_effect_sizes = sibling_1_data.additive_effect_sizes.copy()
        sibling_2_data.causal_loci = sibling_1_data.causal_loci.copy()
        sibling_2_data.epistatic_effects = sibling_1_data.epistatic_effects.copy()

        sibling_2_data.environmental_factors = simulator._generate_environmental_factors()
        sibling_2_data.environmental_effects = simulator._generate_environmental_effects()
        sibling_2_data.environmental_noise = simulator._generate_environmental_noise()

        sibling_2_data.phenotypes = simulator._calculate_phenotypes(params, sibling_2_data)

        # For each phenotype, calculate sibling correlation.
        for phenotype_idx in range(params.num_phenotypes):
            sibling_1_phenotype = sibling_1_data.phenotypes[:, phenotype_idx]
            sibling_2_phenotype = sibling_2_data.phenotypes[:, phenotype_idx]

            # Skip phenotypes with near-zero variance.
            if np.var(sibling_1_phenotype) < 1e-6:
                continue

            sibling_correlation = np.corrcoef(sibling_1_phenotype, sibling_2_phenotype)[0, 1]
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
                    f"but got {correlations[i - 1]} for h2={heritabilities[i - 1]} and "
                    f"{correlations[i]} for h2={heritabilities[i]}"
                )


@pytest.mark.parametrize("env_factors_weight", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("heritability", [0.0, 0.3, 0.6, 1.0])
def test_phenotypic_variance(heritability, env_factors_weight):
    """
    Test that the phenotypic variance is close to 1.0.

    Because the phenotypes are a scaled linear combination of the genetic and environmental
    factors, the variance of the phenotypes should always be 1.0.
    """
    params = SimulationParameters(
        seed=RANDOM_SEED,
        num_samples=10000,
        num_loci=100,
        num_phenotypes=3,
        num_causal_loci_per_phenotype=10,
        pleiotropy_probability=0.0,
        epistasis_probability=0.0,
        env_factors_weight=env_factors_weight,
        # There are no epistatic effects, so the narrow and broad sense heritabilities
        # must be the same.
        broad_sense_heritability=heritability,
        narrow_sense_heritability=heritability,
    )

    simulator = GenotypePhenotypeSimulator(params)
    data = simulator.simulate()

    phenotype_variances = np.var(data.phenotypes, axis=0)

    # The tolerance was chosen empirically.
    assert np.allclose(phenotype_variances, 1.0, atol=0.1)


@pytest.mark.parametrize("epistasis_probability", [0.0, 0.3, 0.6, 1.0])
@pytest.mark.parametrize("broad_sense_heritability", [0.0, 0.3, 0.6, 1.0])
@pytest.mark.parametrize("narrow_sense_heritability_fraction", [0.0, 0.5, 1.0])
def test_phenotypic_variance_with_epistasis(
    epistasis_probability, broad_sense_heritability, narrow_sense_heritability_fraction
):
    """
    Test that the phenotypic variance is always close to 1.0 even in the presence of epistasis
    (i.e., non-additive effects).
    """
    # If there is no epistasis, the narrow-sense heritability must be greater than 0.
    if epistasis_probability == 0.0 and narrow_sense_heritability_fraction < 1:
        pytest.skip(
            "Skipping test because there is no epistasis but broad-sense heritability is greater "
            "than narrow-sense heritability."
        )

    params = SimulationParameters(
        seed=RANDOM_SEED,
        num_samples=10000,
        num_loci=100,
        num_phenotypes=3,
        num_causal_loci_per_phenotype=10,
        pleiotropy_probability=0.0,
        epistasis_probability=epistasis_probability,
        broad_sense_heritability=broad_sense_heritability,
        narrow_sense_heritability=(broad_sense_heritability * narrow_sense_heritability_fraction),
        # No environmental factors; all variance due to env noise.
        env_factors_weight=0.0,
    )

    simulator = GenotypePhenotypeSimulator(params)
    data = simulator.simulate()

    phenotype_variances = np.var(data.phenotypes, axis=0)
    assert np.allclose(phenotype_variances, 1.0, atol=0.1)


@pytest.mark.parametrize("pleiotropy_probability", [0.0, 0.3, 0.6, 1.0])
def test_pleiotropy_rate(pleiotropy_probability):
    """
    Test that the pleiotropy rate is close to the specified probability.
    """
    params = SimulationParameters(
        seed=RANDOM_SEED,
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
    """
    Test that the epistasis rate is close to the specified probability.
    """
    params = SimulationParameters(
        seed=RANDOM_SEED,
        num_samples=5000,
        num_loci=500,
        num_phenotypes=20,
        num_causal_loci_per_phenotype=10,
        epistasis_probability=epistasis_probability,
        # We're only counting epistatic interactions, so we can simplify things by setting
        # the heritability to 0.
        broad_sense_heritability=0.0,
        narrow_sense_heritability=0.0,
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
@pytest.mark.parametrize("effect_size_distribution", ["normal", "exponential", "gamma"])
def test_heritability_with_model_fit(genome_type, heritability, effect_size_distribution):
    """
    Check that the heritability is close to the target value when fitting a linear model
    to simulated data with only additive effects.
    """
    params = SimulationParameters(
        seed=RANDOM_SEED,
        genome_type=genome_type,
        num_samples=1000,
        num_loci=100,
        num_phenotypes=3,
        num_causal_loci_per_phenotype=10,
        pleiotropy_probability=0.0,
        epistasis_probability=0.0,
        broad_sense_heritability=heritability,
        # All heritability is narrow-sense (i.e., there are only additive effects).
        narrow_sense_heritability=heritability,
        effect_size_distribution=effect_size_distribution,
    )

    simulator = GenotypePhenotypeSimulator(params)
    data = simulator.simulate()

    # Fit a linear model to the data.
    # (The value of `alpha` was chosen empirically to ensure the model is not too strict.)
    model = Lasso(alpha=0.03)
    model.fit(data.genotypes, data.phenotypes)

    # Check that the coefficient of determination is close to the target heritability.
    assert np.isclose(model.score(data.genotypes, data.phenotypes), heritability, atol=0.1)


# We don't test with `heritability = 0.0` here because we're only interested
# in the effect of epistasis when there is some heritability to model.
@pytest.mark.parametrize("broad_sense_heritability", [0.3, 0.6, 1.0])
def test_model_fit_with_epistasis(broad_sense_heritability):
    """
    Test that the model fit gets worse as the rate of epistasis increases.
    """

    # Note: the model fit is not super sensitive to the epistasis probability,
    # so we only test three values.
    epistasis_probabilities = [0.0, 0.5, 1.0]

    model_scores = []
    for epistasis_probability in epistasis_probabilities:
        params = SimulationParameters(
            seed=RANDOM_SEED,
            num_samples=1000,
            num_loci=100,
            num_phenotypes=3,
            num_causal_loci_per_phenotype=10,
            pleiotropy_probability=0.0,
            epistasis_probability=epistasis_probability,
            broad_sense_heritability=broad_sense_heritability,
            narrow_sense_heritability=(
                broad_sense_heritability if epistasis_probability == 0 else 0
            ),
            effect_size_distribution="normal",
        )

        simulator = GenotypePhenotypeSimulator(params)
        data = simulator.simulate()

        model = Lasso(alpha=0.03)
        model.fit(data.genotypes, data.phenotypes)
        model_score = model.score(data.genotypes, data.phenotypes)
        model_scores.append(model_score)

        # If there is epistasis present, the model fit should be worse than the heritability.
        if epistasis_probability > 0.0:
            assert model_score < broad_sense_heritability
        else:
            assert np.isclose(model_score, broad_sense_heritability, atol=0.1)

    # The model fit should get worse as the extent of epistasis increases.
    for ind in range(1, len(model_scores)):
        assert model_scores[ind] < model_scores[ind - 1]


@pytest.mark.parametrize("heritability", [0.0, 0.3, 0.6, 1.0])
@pytest.mark.parametrize("pleiotropy_probability", [0.0, 0.5, 1.0])
def test_model_fit_with_pleiotropy(heritability, pleiotropy_probability):
    """
    Test that the model fit does not depend on the pleiotropy rate.

    Pleiotropic interactions are still linear, so a linear model should fit the data just as well
    with or without pleiotropy.
    """
    params = SimulationParameters(
        seed=RANDOM_SEED,
        num_samples=1000,
        num_loci=100,
        num_phenotypes=3,
        num_causal_loci_per_phenotype=10,
        pleiotropy_probability=pleiotropy_probability,
        epistasis_probability=0.0,
        broad_sense_heritability=heritability,
        # All heritability is narrow-sense (i.e., there are only additive effects).
        narrow_sense_heritability=heritability,
        effect_size_distribution="normal",
    )

    simulator = GenotypePhenotypeSimulator(params)
    data = simulator.simulate()

    model = Lasso(alpha=0.03)
    model.fit(data.genotypes, data.phenotypes)
    model_score = model.score(data.genotypes, data.phenotypes)

    # The model fit should be comparable to the heritability, regardless of pleiotropy.
    assert np.isclose(model_score, heritability, atol=0.1)


def test_data_to_hdf5(tmp_path):
    """
    Test that the data can be saved to a file correctly.
    """
    num_samples = 100
    num_loci = 50
    num_phenotypes = 3

    params = SimulationParameters(
        seed=RANDOM_SEED,
        num_samples=num_samples,
        num_loci=num_loci,
        num_phenotypes=num_phenotypes,
    )
    simulator = GenotypePhenotypeSimulator(params)
    data = simulator.simulate()
    data.to_hdf5(tmp_path / "data.h5", one_hot_encode_genotypes=False)

    with h5py.File(tmp_path / "data.h5", "r") as file:
        expected_params = json.loads(json.dumps(params.model_dump()))
        observed_params = json.loads(file["metadata/params"][...].item().decode())  # type: ignore
        assert observed_params == expected_params

        assert file["arrays/genotypes"].shape == (num_samples, num_loci)  # type: ignore
        assert file["arrays/phenotypes"].shape == (num_samples, num_phenotypes)  # type: ignore
