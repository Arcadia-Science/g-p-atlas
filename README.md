# G-P Atlas

- [G-P Atlas](#g-p-atlas)
  - [Overview](#overview)
  - [Usage](#usage)
    - [Dataset format](#dataset-format)
    - [Demo dataset](#demo-dataset)
  - [Usage on Modal](#usage-on-modal)
  - [Simulation scripts](#simulation-scripts)
  - [Repo directory structure](#repo-directory-structure)
  - [Development](#development)
    - [Environment setup](#environment-setup)
    - [Testing](#testing)
    - [Managing dependencies](#managing-dependencies)
    - [Formatting and linting](#formatting-and-linting)
    - [Pre-commit hooks](#pre-commit-hooks)

## Overview

This repo contains a Python package called `gpatlas` that is a more flexible and robust implementation of the G-P Atlas model discussed in the [G-P Atlas pub](https://research.arcadiascience.com/pub/result-g-p-atlas). This model was originally implemented in scripts found in the [2025-g-p-atlas](https://github.com/Arcadia-Science/2025-g-p-atlas) repo. The package in this repo reorganizes the functionality in those scripts to make it more modular and reusable. It also adds a command-line interface for training and evalulating models locally and on [Modal](https://modal.com/). Finally, it adds unit tests for most of the major features.

## Usage

This package does not have a formal versioned release or a release on PyPI. To install it, follow the directions in the ["Development"](#development) section below. Once the package is installed, the `gpatlas` CLI will be available. To train a model, use the `train` command:

```bash
gpatlas train \
    --dataset /path/to/dataset/directory \
    --output /path/to/output/directory \
    --model-config /path/to/model-config.yaml \
    --train-config /path/to/trainer-config.yaml
```

This command will train a model and save the best model checkpoint and the training logs in the output directory. It will automatically select the best device to use, but will of course be much faster if a GPU is available.

The `--dataset` directory should contain two HDF5 files: `train-data.h5` and `test-data.h5`, with the training data and test data, respectively. See below for more information on the format of these files.

The `model-config.yaml` and `trainer-config.yaml` files should specify the model architecture and the training configuration. The schema for these config files is defined by the pydantic models in [`gpatlas/models.py`](gpatlas/models.py) and [`gpatlas/trainers.py`](gpatlas/trainers.py), respectively.

The `--model-config` and `--trainer-config` flags can be ommitted if `model-config.yaml` and `trainer-config.yaml` are located in the dataset directory.

To evaluate a trained model, use the `evaluate` command:

```bash
gpatlas evaluate \
    --dataset /path/to/dataset/directory \
    --output /path/to/output/directory
```

This will generate plots in the `plots/` directory and save the predicted phenotypes in the `predictions/` directory of the output directory.

After training and evaluating a model, the output directory will contain the following files and directories:

```
/path/to/output/directory
├── checkpoints
├── logs
├── plots
├── predictions
├── model-config.yaml
└── train-config.yaml
```


### Dataset format

The dataset directory should contain two HDF5 files: `train-data.h5` and `test-data.h5`, containing the training data and test data, respectively. These files should both contain two datasets: either `genotypes` and `phenotypes` or `arrays/genotypes` and `arrays/phenotypes`.

The `genotypes` dataset should be an array of one-hot encoded allelic states for each sample and locus. Its shape should be `(num_samples, num_loci, num_allelic_states)`. In most cases, `num_allelic_states` is either 2 for bi-allelic loci or 3 for bi-allelic loci in a diploid organism.

The `phenotypes` dataset should be an array of quantitative phenotypes for each sample with shape `(num_samples, num_phenotypes)`.

The order of the samples in the first dimension of the two datasets is assumed to match.

### Demo dataset

This repo includes a small demo dataset in the `demo/` directory. See the [demo README](demo/README.md) for an example of how to train a model on this dataset.

## Usage on Modal

We use [Modal](https://modal.com/) to conveniently run the `gpatlas` commands on GPU nodes. This requires a Modal account. See the [Modal docs](https://modal.com/docs/guide) for more information on how to set up the Modal CLI.

The training datasets and model checkpoints are stored in an S3 bucket that is mounted as a Modal volume. To set up the S3 bucket, copy the `.env.copy` file to `.env` and fill in the variables. The AWS access credentials in the `.env` file must have full access to the S3 bucket named in the `.env` file, and the bucket must already exist.

The `gpatlas-modal` CLI is identical to the main `gpatlas` CLI documented above, except that the underlying commands are run on Modal. For example, to train a model on a GPU, run the following command:

```bash
gpatlas-modal train \
    --dataset /volume/some-dataset \
    --output /volume/some-output
```

Here, `/volume` is the directory where the S3 bucket is mounted. The path `/volume/some-dataset` will correspond to the path `s3://<your-bucket-name>/some-dataset` in the S3 bucket (where `<your-bucket-name>` is the name of the S3 bucket specified in the `.env` file).

## Simulation scripts

The repo contains two scripts for simulating genotype-phenotype data.

- [`scripts/pub_gp_simulation.py`](scripts/pub_gp_simulation.py): This script is a refactored version of the simulation script `tools_for_phen_gen_creation.py` used to generate the simulated genotype-phenotype dataset in the [G-P Atlas pub](https://research.arcadiascience.com/pub/result-g-p-atlas).

- [`gpatlas/simulations.py`](gpatlas/simulations.py): This script simulates genotype-phenotype data under a wider but still simplistic range of conditions. It is also more robustly tested (in [`gpatlas/tests/test_simulations.py`](gpatlas/tests/test_simulations.py)). Note that datasets generated by this script are not comparable to datasets generated by the `pub_gp_simulation.py` script above, as it takes a different approach to simulating epistasis, pleiotropy, and environmental effects.

## Repo directory structure

This repo is organized as follows:

```
g-p-atlas/
├── external/    # Snapshots of scripts from the original 2025-g-p-atlas repo (for reference).
├── scripts/     # Standalone scripts useful for troubleshooting and development.
└── gpatlas/     # The `gpatlas` package.
    ├── tests/           # Unit tests.
    ├── cli.py           # Main command-line interface for the `gpatlas` package.
    ├── constants.py     # Constants used throughout the package.
    ├── datasets.py      # Dataset classes.
    ├── layers.py        # Custom PyTorch layers.
    ├── modal_cli.py     # Command-line interface for using the `gpatlas` package on Modal.
    ├── models.py        # The main G-P Atlas model.
    ├── simulations.py   # A simulation script for generating genotype-phenotype data.
    ├── trainers.py      # Model training and evaluation loops.
    └── utils.py
```

## Development

### Environment setup

We use [`uv`](https://docs.astral.sh/uv/) for dependency management. First, [install `uv`](https://docs.astral.sh/uv/getting-started/installation/) globally.

Next, install dependencies:

```bash
uv sync --all-groups
```

It is convenient to work within the `uv` virtual environment. To do so, run the following command:

```bash
source .venv/bin/activate
```

Now, the `gpatlas` CLI will be available. See above for usage instructions.

### Testing

We use `pytest` for testing. The tests are found in the `gpatlas/tests/` subpackage. To run the tests, use the following command:

```bash
make test
```

### Managing dependencies

To add a new dependency, use the following command:

```bash
uv add some-package
```

To add a new development dependency, use the following command:

```bash
uv add --dev some-dev-package
```

To update a dependency, use the following command:

```bash
uv add --upgrade some-package
```

To update a development dependency, use the following command:

```bash
uv add --dev --upgrade some-dev-package
```

Whenever you add or update a dependency, `uv` will automatically update both `pyproject.toml` and the `uv.lock` file. Make sure to commit the changes to **both** of these files to the repo.

### Formatting and linting

To format the code, use the following command:

```bash
make format
```

To run the lint checks and type checking, use the following command:

```bash
make lint
```

### Pre-commit hooks

We use pre-commit to run formatting and lint checks before each commit. To install the pre-commit hooks, use the following command:

```bash
pre-commit install
```

To run the pre-commit checks manually, use the following command:

```bash
make pre-commit
```
