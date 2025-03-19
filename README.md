
# g-p-atlas

This repo contains a Python package called `g-p-atlas`.

## TODO: Installation

## TODO: Usage

## Development

### Environment setup

We use poetry for dependency management and build tooling. You can either install poetry globally or within a virtual environment in order to isolate poetry itself. We recommend the latter. First, create a new conda environment from the `dev.yml` environment file:

```bash
conda env create -n g-p-atlas-dev -f envs/dev.yml
conda activate g-p-atlas-dev
```

Next, install dependencies, including the development dependencies:

```bash
poetry install --no-root --with dev
```

If you have installed poetry globally, activate the poetry virtual environment:

```bash
poetry shell
```

Poetry detects and respects existing virtual environments, so if you are using poetry within a conda environment, this step is not needed.

Finally, install the package itself in editable mode:

```bash
pip install -e .
```

### Formatting and linting

If you have installed poetry globally, make sure to run `poetry shell` before running the commands below.

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

### Testing

We use `pytest` for testing. The tests are found in the `g_p_atlas/tests/` subpackage. To run the tests, use the following command:

```bash
make test
```

### Managing dependencies

We use poetry to manage dependencies. To add a new dependency, use the following command:

```bash
poetry add some-package
```

To add a new development dependency, use the following command:

```bash
poetry add -G dev some-dev-package
```

To update a dependency, use the following command:

```bash
poetry update some-package
```

Whenever you add or update a dependency, poetry will automatically update both `pyproject.toml` and the `poetry.lock` file. Make sure to commit the changes to these files to the repo.
