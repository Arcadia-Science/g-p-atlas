import pathlib
from typing import Any

import click
import modal

from gpatlas.cli import evaluate_command, train_command
from gpatlas.utils import find_package_root_dir

GPU_NAME = "a10g"
CPU_COUNT = 4
MEMORY_MB = 1024 * 10
TIMEOUT_SECONDS = 3600
S3_BUCKET_NAME = "2025-kcc-gpatlas"
VOLUME_PATH = pathlib.Path("/volume")

# Note: this root directory only needs to be correct when executing this module locally, not when
# this module is executed on Modal (where there is no local repo), but it always needs to be
# a valid `pathlib.Path` object.
PACKAGE_ROOT_DIR = find_package_root_dir() or pathlib.Path("")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .add_local_file(
        PACKAGE_ROOT_DIR / "pyproject.toml",
        remote_path="/app/pyproject.toml",
        copy=True,
    )
    .run_commands(
        "pip install poetry",
        # Configure poetry to install dependencies in the system Python environment.
        "poetry config virtualenvs.create false",
    )
    # Install dependencies in a separate command to optimize caching.
    .run_commands(
        "cd /app && poetry install --no-root",
    )
    .add_local_python_source("gpatlas")
)

app = modal.App(name="2025-kcc-gpatlas")


def app_function(**kwargs: Any):
    """
    Create an `app.function` decorator with the given kwargs.
    """
    default_kwargs = {
        "cpu": CPU_COUNT,
        "gpu": GPU_NAME,
        "memory": MEMORY_MB,
        "timeout": TIMEOUT_SECONDS,
    }
    default_kwargs.update(kwargs)

    s3_bucket_mount = modal.CloudBucketMount(S3_BUCKET_NAME, secret=modal.Secret.from_dotenv())

    return app.function(
        image=image,
        volumes={str(VOLUME_PATH): s3_bucket_mount},
        **default_kwargs,
    )


def run_command_on_modal(command_name: str, command_kwargs: dict[str, Any]):
    """
    Run a CLI command on modal.

    This function is intended to be decorated with an `app.function` decorator.
    """
    # Import the original CLI commands on modal.
    from gpatlas.cli import evaluate_command, train_command

    commands = {
        "train": train_command,
        "evaluate": evaluate_command,
    }

    command = commands[command_name]
    ctx = click.Context(command)
    ctx.invoke(command, **command_kwargs)


@click.group()
def cli():
    pass


@click.command()
@click.option("--gpu", type=str, default=GPU_NAME)
def modal_train_command(gpu: str, **kwargs: Any):
    _run_command_on_modal = app_function(gpu=gpu)(run_command_on_modal)
    with modal.enable_output(), app.run():
        _run_command_on_modal.remote("train", kwargs)


@click.command()
@click.option("--gpu", type=str, default=GPU_NAME)
def modal_evaluate_command(gpu: str, **kwargs: Any):
    _run_command_on_modal = app_function(gpu=gpu)(run_command_on_modal)
    with modal.enable_output(), app.run():
        _run_command_on_modal.remote("evaluate", kwargs)


# Append the CLI parameters from the original commands.
modal_train_command.params += train_command.params
modal_evaluate_command.params += evaluate_command.params

cli.add_command(modal_train_command, name="train")
cli.add_command(modal_evaluate_command, name="eval")


if __name__ == "__main__":
    cli()
