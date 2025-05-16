import numpy as np
import copy as copy
import pickle as pk


def make_genotype(
    n_as=None,
    n_loci=None,
    n_loci_ip=None,
    n_env=None,
    n_animals=None,
    n_phens=None,
    env_weight=None,
    p_interact=None,
    p_pleio=None,
    noise=None,
    downsample=None,
):
    """a simple tool for generating phenotypic and genetic data.  Currently, this
    allows for the addition of non-linear gene-gene interactions, but the model is limited.
    An allele of a gene can only interact with one other allele of another gene. This now
    allows for pleiotropy.  For any individual gene known to influence a phenotype, there is a
    p_pleio probability that it will also influence any other individual phenotype.  The degree
    of influence is set by p_pleio. Creates a training output with n_animals number of animals and a
    testing output with 0.2*n_animals number of animals.

    n_as is the number of allelic states at all segregating loci
    n_loci is the number of segregating loci in the analysis
    n_loci_ip is the number of loci influencing the phenotype
    n_env is the number of differing environmental influences on a phenotype
    n_animals is the number of animals to be simulated
    n_phens is the number of phenotypes to be simulated
    env_weight is the strength of environmental influence on the phenotype
    p_interact is the probability of interaction between two loci
    p_pleio is the probability of pleiotoropy between a locus-phenotype pair
    noise is the amount of noise added
    """

    if n_as == None:
        n_as = 3
    if n_loci == None:
        n_loci = 3000
    if n_loci_ip == None:
        n_loci_ip = 10
    if n_animals == None:
        n_animals = 600
    if n_phens == None:
        n_phens = 30
    if p_interact == None:
        p_interact = 0.1
    if p_pleio == None:
        p_pleio = 0.1
    if n_env == None:
        n_env = 2
    if env_weight == None:
        env_weight = 0.2
    if noise == None:
        noise = 0.01
    if downsample == None:
        downsample = False

    # make a dictionary for output and add all metadata
    out_dct = {}
    out_dct["n_as"] = n_as
    out_dct["n_loci"] = n_loci
    out_dct["n_loci_ip"] = n_loci_ip
    out_dct["n_animals"] = n_animals
    out_dct["n_phens"] = n_phens
    out_dct["p_interact"] = p_interact
    out_dct["p_pleio"] = p_pleio
    out_dct["n_env"] = n_env
    out_dct["env_weight"] = env_weight
    out_dct["noise"] = noise
    out_dct["downsample"] = downsample

    # set up n_animals for test and train data sets
    n_animals_train = n_animals
    n_animals = int(n_animals * 1.2)
    n_animals_test = n_animals - n_animals_train

    # set random seed
    np.random.seed(47)

    # make weights for genotypes
    inds = np.zeros((n_phens, n_loci_ip), dtype=int).tolist()
    weights = np.zeros((n_phens, n_loci, n_as))
    for n in range(n_loci_ip):
        for m in range(n_phens):
            ind = int(np.random.randint(n_loci))
            weights[m][ind] = np.random.rand(n_as) * 10
            inds[m][n] = int(ind)

    # add pleiotropy.  For each locus where there is influence on any one phenotype, this
    # will add an influence on each other phenotype with probability p_pleio.
    all_inds = copy.deepcopy(inds)
    pleiotropy_mat = []
    for m in range(n_phens):
        gens_mat = []
        for n in list(range(0, m)) + list(range(m + 1, n_phens)):
            for z in inds[m]:
                if np.random.binomial(1, p_pleio):
                    # weights[n][z]=np.random.rand(n_as)*10
                    weights[n][z] = weights[m][z]
                    all_inds[n].append(z)
                    gens_mat.append([m, n, z])
        pleiotropy_mat.append(gens_mat)

    # reduce number of loci influencing a phenotype to the number stated in n_loci_ip even if there are pleiotropic effects
    new_inds = list(range(n_phens))
    if downsample == True:
        new_weights = np.zeros((n_phens, n_loci, n_as)).tolist()
        for m in range(n_phens):
            indss = np.random.choice(all_inds[m], n_loci_ip, replace=False)
            new_inds[m] = indss
            for ind in indss:
                new_weights[m][ind] = weights[m][ind]
        weights = np.array(new_weights)
        inds = np.array(new_inds)

    # make genotypes
    genotypes, gen_locs = zip(*[make_genotype_ind(n_as, n_loci) for x in range(n_animals)])

    # make weighted genotypes
    weighted_genes = []
    for n in range(n_phens):
        weighted_genes.append(genotypes * weights[n])

    # make gene-gene interactions
    interact = []
    interacting_loci = []
    for n in range(n_phens):
        phen_interact = []
        for m in range(n_loci_ip):
            for z in range(m + 1, n_loci_ip):
                if np.random.binomial(1, p_interact):
                    ind_1 = inds[n][m]
                    ind_2 = inds[n][z]
                    if ind_1 not in interacting_loci and ind_2 not in interacting_loci:
                        interacting_loci.append(ind_1)
                        interacting_loci.append(ind_2)
                        allele = np.random.randint(3)
                        phen_interact.append([ind_1, ind_2, allele])
        interact.append(phen_interact)

    # impose gene-gene interactions
    for n in range(n_phens):
        for m in range(n_animals):
            for intr in interact[n]:
                loc_1 = weighted_genes[n][m][intr[0]][intr[2]]
                loc_2 = weighted_genes[n][m][intr[1]][intr[2]]
                if loc_1 != 0 and loc_2 != 0:
                    new_weight = loc_1 * loc_2
                    weighted_genes[n][m][intr[0]][intr[2]] = new_weight
                    weighted_genes[n][m][intr[1]][intr[2]] = 0

    # calculate individual phenotypes
    phens = []
    for n in range(n_phens):
        phens.append(np.sum(weighted_genes[n], axis=(2, 1)))

    # add environmental effects
    env_phens = copy.deepcopy(phens)

    # add noise
    noise_vects = np.random.rand(n_phens, n_animals)
    noise_vects = noise_vects * (np.mean(env_phens) / np.mean(noise_vects)) * noise
    noisy_phens = phens + noise_vects

    # format data for output
    out_dct["weights"] = weights
    out_dct["inds_of_loci_influencing_phen"] = inds
    out_dct["interact_matrix"] = interact
    out_dct["pleiotropy_matrix"] = pleiotropy_mat

    test_out_dct = out_dct.copy()

    # append genetic data
    out_dct["genotypes"] = genotypes[:n_animals_train]
    out_dct["gen_locs"] = gen_locs[:n_animals_train]

    test_out_dct["n_animals"] = n_animals_test
    test_out_dct["genotypes"] = genotypes[n_animals_train:]
    test_out_dct["gen_locs"] = gen_locs[n_animals_train:]

    # re-structure phenotype data and append
    phens = list(np.array(phens).T)
    noisy_phens = list(np.array(noisy_phens).T)
    env_phens = list(np.array(env_phens).T)

    out_dct["phens"] = phens[:n_animals_train]
    out_dct["noisy_phens"] = noisy_phens[:n_animals_train]
    out_dct["env_phens"] = env_phens[:n_animals_train]

    test_out_dct["phens"] = phens[n_animals_train:]
    test_out_dct["noisy_phens"] = noisy_phens[n_animals_train:]
    test_out_dct["env_phens"] = env_phens[n_animals_train:]

    # return genotypes,gen_locs,weights,phens,inds,interact,pleiotropy_mat
    return out_dct, test_out_dct


def make_genotype_ind(n_as, n_loci):
    """utlity function for make_genotype.  Creates a random set of genotypes
    for an individual."""
    gen = np.zeros((n_loci, n_as))
    locs = []
    for n in range(n_loci):
        loc = np.random.randint(n_as)
        gen[n][loc] = 1
        locs.append(loc)
    return gen, locs


def p_i_sweep(outpath):
    """utility function that creates a 2d sweep through probability of peleiotropy and probability of interaction"""
    incr = np.array(range(0, 11, 1)) / 10
    for i in incr:
        for f in incr:
            train, test = make_genotype(
                downsample=True, p_pleio=i, p_interact=f, n_animals=3000, noise=0
            )
            pk.dump(
                train,
                open(outpath + "train_pleio_" + str(i) + "_int_" + str(f) + ".pk", "wb"),
            )
            pk.dump(
                test,
                open(outpath + "test_pleio_" + str(i) + "_int_" + str(f) + ".pk", "wb"),
            )


def convert_p_i_sweep(list_of_files):
    out_phens = []
    incr = np.array(range(0, 102, 2)) / 100
    for i in incr:
        int_inc = []
        for f in incr:
            filname = [x for x in list_of_files if "pleio_" + str(i) + "_int_" + str(f) in x][0]
            dat = pk.load(open(filname, "br"))
            int_inc.append(dat["phens"])
        out_phens.append(int_inc)
