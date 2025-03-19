import copy
import pickle
from typing import Any

import numpy as np
from numpy.typing import NDArray


def simulate_data(
    num_allelic_states: int = 3,
    num_total_loci: int = 3000,
    num_active_loci: int = 10,
    num_samples: int = 600,
    num_phenotypes: int = 30,
    interaction_probability: float = 0.1,
    pleiotropy_probability: float = 0.1,
    noise_strength: float = 0.01,
    downsample: bool = False,
    seed: int = 47,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Simulate phenotypic and genetic data.

    This function allows for the addition of gene-gene interactions and pleiotropy as follows:
    - An allele of one gene can only interact with one other allele of another gene.
    - Any individual gene known to influence a phenotype has a chance
      to influence any other individual phenotype given by `pleiotropy_probability`.

    This function outputs two dictionaries:
    - A training output with `num_samples` number of samples.
    - A testing output with `num_samples * 0.2` number of samples.

    Arguments:
    `num_allelic_states` is the number of allelic states at all segregating loci
    `num_total_loci` is the number of segregating loci in the simulation.
    `num_active_loci` is the number of loci influencing a single phenotype
    `num_samples` is the number of samples (e.g. animals or individuals) to be simulated.
    `num_phenotypes` is the number of phenotypes to be simulated.
    `interaction_probability` is the probability of interaction between two loci
    `pleiotropy_probability` is the probability of pleiotoropy between a locus-phenotype pair
    `noise_strength` is the amount of noise added. TODO (KC) added to what?
    `downsample` is a boolean flag. TODO (KC) what does this do?
    """

    np.random.seed(seed)

    data = {}
    data["num_allelic_states"] = num_allelic_states
    data["num_total_loci"] = num_total_loci
    data["num_active_loci"] = num_active_loci
    data["num_samples"] = num_samples
    data["num_phenotypes"] = num_phenotypes
    data["interaction_probability"] = interaction_probability
    data["pleiotropy_probability"] = pleiotropy_probability
    data["noise_strength"] = noise_strength
    data["downsample"] = downsample

    num_samples_train = num_samples
    num_samples_test = int(num_samples * 0.2)

    # Lists of indices of 'active' loci for each phenotype.
    # (The 'active' loci are the loci that affect each phenotype.)
    active_loci_inds: list[list[int]] = np.zeros(
        (num_phenotypes, num_active_loci), dtype=int
    ).tolist()  # type: ignore

    # Matrix of allele weights for each phenotype and locus.
    # These determine how much each allele at each locus contributes to each phenotype.
    weights = np.zeros((num_phenotypes, num_total_loci, num_allelic_states))

    for phenotype_ind in range(num_phenotypes):
        for active_locus_ind in range(num_active_loci):
            # Pick a random locus index to be an active locus for this phenotype.
            # TODO (KC): shouldn't this sample without replacement?
            locus_ind = int(np.random.randint(num_total_loci))

            # Set the weight of each allele at the locus to a random value between 0 and 10.
            weights[phenotype_ind][locus_ind] = np.random.rand(num_allelic_states) * 10

            # Assign the locus index to the current pair of phenotype and active locus.
            active_loci_inds[phenotype_ind][active_locus_ind] = locus_ind

    # add pleiotropy.  For each active locus, this will add an effect on each other phenotype
    # with probability `pleiotropy_probability`.
    all_active_loci_inds = copy.deepcopy(active_loci_inds)

    # List of pleiotropic effects. This is not a matrix because the number of pleiotropic effects
    # is different for each phenotype.
    # The "shape" is (num_phenotypes, num_pleiotropic_effects_per_phenotype, 3).
    # The tuples are (phenotype_1_ind, phenotype_2_ind, locus_ind).
    all_pleiotropic_effects: list[list[tuple[int, int, int]]] = []

    for phenotype_ind in range(num_phenotypes):
        pleiotropic_effects: list[tuple[int, int, int]] = []
        for other_phenotype_ind in range(num_phenotypes):
            if other_phenotype_ind == phenotype_ind:
                continue
            for active_locus_ind in active_loci_inds[phenotype_ind]:
                if np.random.binomial(1, pleiotropy_probability):
                    weights[other_phenotype_ind][active_locus_ind] = weights[phenotype_ind][
                        active_locus_ind
                    ].copy()

                    all_active_loci_inds[other_phenotype_ind].append(active_locus_ind)
                    pleiotropic_effects.append(
                        (phenotype_ind, other_phenotype_ind, active_locus_ind)
                    )
        all_pleiotropic_effects.append(pleiotropic_effects)

    # Randomly downsample the active loci per phenotype to `num_active_loci`.
    # This is necessary when there are pleiotropic effects, because the number of active loci
    # per phenotype will be greater than `num_active_loci`.
    if downsample:
        downsampled_active_loci_inds: list[np.ndarray] = []
        downsampled_weights = np.zeros(
            (num_phenotypes, num_total_loci, num_allelic_states)
        ).tolist()
        for phenotype_ind in range(num_phenotypes):
            downsampled_inds = np.random.choice(
                all_active_loci_inds[phenotype_ind], num_active_loci, replace=False
            )
            downsampled_active_loci_inds.append(downsampled_inds)  # type: ignore
            for downsampled_ind in downsampled_inds:
                downsampled_weights[phenotype_ind][downsampled_ind] = weights[phenotype_ind][
                    downsampled_ind
                ].copy()

        # Reassign the downsampled weights and active loci indices to the original variables.
        # TODO (KC): I belive there is a bug here, as the downsampled indices are derived from
        # `all_active_loci_inds`, but if we don't downsample, then `active_loci_inds`
        # is never updated to point to `all_active_loci_inds`.
        weights = np.array(downsampled_weights)
        active_loci_inds = np.array(downsampled_active_loci_inds).tolist()  # type: ignore

    def make_sample_genotype(
        num_allelic_states: int, num_loci: int
    ) -> tuple[np.ndarray, list[int]]:
        """
        Creates a genotype for an individual by randomly assigning an allele to each locus.
        This function returns two representations of the genotype:
        - A one-hot encoded matrix, where each row corresponds to a locus
          and each column corresponds to an allele.
        - A list of allele indices, where each element corresponds to the index of the allele
          at that locus.
        """
        genotype_one_hot: np.ndarray = np.zeros((num_loci, num_allelic_states))
        genotype_allele_inds: list[int] = []

        for locus_ind in range(num_loci):
            allele_ind = np.random.randint(num_allelic_states)
            genotype_one_hot[locus_ind][allele_ind] = 1
            genotype_allele_inds.append(allele_ind)
        return genotype_one_hot, genotype_allele_inds

    # Create genotypes for all samples (e.g. animals).
    genotypes_as_one_hot: list[NDArray[np.float64]] = []
    genotypes_as_allele_inds: list[list[int]] = []
    for _ in range(num_samples):
        genotype_one_hot_for_sample, genotype_allele_inds_for_sample = make_sample_genotype(
            num_allelic_states, num_total_loci
        )
        genotypes_as_one_hot.append(genotype_one_hot_for_sample)
        genotypes_as_allele_inds.append(genotype_allele_inds_for_sample)

    # TODO (KC): Avoid re-casting to numpy arrays.
    genotypes_as_one_hot = np.array(genotypes_as_one_hot)  # type: ignore

    # Create weighted genotypes (these are later summed over alleles and loci to yield phenotypes).
    weighted_genotypes: list[NDArray[np.float64]] = []
    for phenotype_ind in range(num_phenotypes):
        weighted_genotypes.append(genotypes_as_one_hot * weights[phenotype_ind])

    # TODO (KC): Avoid re-casting to numpy arrays.
    weighted_genotypes = np.array(weighted_genotypes)  # type: ignore

    # make gene-gene interactions
    interacting_locus_inds: list[int] = []
    gene_gene_interactions: list[list[tuple[int, int, int]]] = []

    for phenotype_ind in range(num_phenotypes):
        phenotype_gene_gene_interactions: list[tuple[int, int, int]] = []

        for active_locus_ind in range(num_active_loci):
            # Interactions are symmetric, so we only need to consider interactions between
            # active loci with indices greater than the current index.
            for other_active_locus_ind in range(active_locus_ind + 1, num_active_loci):
                if np.random.binomial(1, interaction_probability):
                    # TODO (KC): I suspect there is a bug here: we are using `active_loci_inds`,
                    # not `all_active_loci_inds`, and these will not be the same
                    # if we didn't downsample.
                    ind_1 = active_loci_inds[phenotype_ind][active_locus_ind]
                    ind_2 = active_loci_inds[phenotype_ind][other_active_locus_ind]
                    if ind_1 not in interacting_locus_inds and ind_2 not in interacting_locus_inds:
                        interacting_locus_inds.append(ind_1)
                        interacting_locus_inds.append(ind_2)
                        # KC: a magic number `3` was used here in the original script;
                        # I think this was a mistake and it should be `num_allelic_states`.
                        allele = np.random.randint(num_allelic_states)
                        phenotype_gene_gene_interactions.append((ind_1, ind_2, allele))

        gene_gene_interactions.append(phenotype_gene_gene_interactions)

    # impose gene-gene interactions
    for sample_ind in range(num_samples):
        for phenotype_ind in range(num_phenotypes):
            for interaction in gene_gene_interactions[phenotype_ind]:
                locus_1_ind, locus_2_ind, allele = interaction
                loc_1 = weighted_genotypes[phenotype_ind][sample_ind][locus_1_ind][allele]
                loc_2 = weighted_genotypes[phenotype_ind][sample_ind][locus_2_ind][allele]
                if loc_1 != 0 and loc_2 != 0:
                    new_weight = loc_1 * loc_2
                    weighted_genotypes[phenotype_ind][sample_ind][locus_1_ind][allele] = new_weight
                    # TODO (KC): why is this set to 0?
                    weighted_genotypes[phenotype_ind][sample_ind][locus_2_ind][allele] = 0

    # Calculate individual phenotypes by summing the weighted genotypes over loci and alleles.
    phenotypes: list[NDArray[np.float64]] = []
    for phenotype_ind in range(num_phenotypes):
        phenotypes.append(np.sum(weighted_genotypes[phenotype_ind], axis=(2, 1)))

    # Add noise to the phenotypes.
    noise_matrix = np.random.rand(num_phenotypes, num_samples)
    noise_matrix = noise_matrix * (np.mean(phenotypes) / np.mean(noise_matrix)) * noise_strength
    noisy_phenotypes = phenotypes + noise_matrix

    # Transpose the phenotype arrays so that their shape is (num_samples, num_phenotypes).
    phenotypes = list(np.array(phenotypes).T)
    noisy_phenotypes = list(np.array(noisy_phenotypes).T)

    data["weights"] = weights
    data["inds_of_loci_influencing_phen"] = active_loci_inds
    data["interact_matrix"] = gene_gene_interactions
    data["pleiotropy_matrix"] = all_pleiotropic_effects

    data["num_samples"] = num_samples_train
    data["genotypes_as_one_hot"] = genotypes_as_one_hot[:num_samples_train]
    data["genotypes_as_allele_inds"] = genotypes_as_allele_inds[:num_samples_train]
    data["phenotypes"] = phenotypes[:num_samples_train]
    data["noisy_phenotypes"] = noisy_phenotypes[:num_samples_train]

    test_data = data.copy()
    test_data["num_samples"] = num_samples_test
    test_data["genotypes_as_one_hot"] = genotypes_as_one_hot[num_samples_train:]
    test_data["genotypes_as_allele_inds"] = genotypes_as_allele_inds[num_samples_train:]
    test_data["phenotypes"] = phenotypes[num_samples_train:]
    test_data["noisy_phenotypes"] = noisy_phenotypes[num_samples_train:]

    return data, test_data


def pleiotropy_interaction_sweep(outpath: str):
    """
    Utility function that creates a 2d sweep through peleiotropy and interaction probabilities.
    """
    step_size = 0.1
    probabilities = np.arange(0, 1.0 + step_size, step_size, dtype=float)

    for pleiotropy_probability in probabilities:
        for interaction_probability in probabilities:
            train_data, test_data = simulate_data(
                pleiotropy_probability=pleiotropy_probability,
                interaction_probability=interaction_probability,
                num_samples=3000,
                noise_strength=0,
                downsample=True,
            )
            pickle.dump(
                train_data,
                open(
                    outpath
                    + "train_data_pleio_"
                    + str(pleiotropy_probability)
                    + "_interaction_"
                    + str(interaction_probability)
                    + ".pk",
                    "wb",
                ),
            )
            pickle.dump(
                test_data,
                open(
                    outpath
                    + "test_data_pleio_"
                    + str(pleiotropy_probability)
                    + "_interaction_"
                    + str(interaction_probability)
                    + ".pk",
                    "wb",
                ),
            )
