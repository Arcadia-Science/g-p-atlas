import pathlib
from typing import Any

import click
import modal

from gpatlas.cli.gpatlas import evaluate_command, train_command


def find_package_root_dir() -> pathlib.Path:
    """
    Find the nearest package root directory.

    If no package root is found, return the directory of this file.
    """
    starting_dir = pathlib.Path(__file__).parent
    current_dir = starting_dir
    while not current_dir.joinpath("pyproject.toml").exists():
        current_dir = current_dir.parent
        if current_dir == pathlib.Path("/"):
            return starting_dir
    return current_dir


GPU_NAME = "a10g"
VOLUME_NAME = "2025-kcc-gpatlas"
VOLUME_PATH = pathlib.Path("/volume")

volume = modal.Volume.from_name(VOLUME_NAME)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .add_local_file(
        find_package_root_dir() / "pyproject.toml",
        remote_path="/app/pyproject.toml",
        copy=True,
    )
    .run_commands(
        "pip install poetry",
        # Install dependencies in the system Python environment.
        "poetry config virtualenvs.create false",
    )
    .run_commands(
        "cd /app && poetry install --no-root",
    )
    .add_local_python_source("gpatlas")
)

app = modal.App(name="2025-kcc-gpatlas")


@app.function(
    image=image,
    cpu=1,
    gpu=GPU_NAME,
    memory=1024 * 10,
    timeout=3600,
    volumes={str(VOLUME_PATH): volume},
)
def run_command_on_modal(command_name: str, command_kwargs: dict[str, Any]):
    """
    Run a CLI command on modal.
    """
    # Import the original CLI commands on modal.
    from gpatlas.cli.gpatlas import evaluate_command, train_command

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
def modal_train_command(**kwargs: Any):
    with modal.enable_output(), app.run():
        run_command_on_modal.remote("train", kwargs)


@click.command()
def modal_evaluate_command(**kwargs: Any):
    with modal.enable_output(), app.run():
        run_command_on_modal.remote("evaluate", kwargs)


# Copy the CLI parameters from the original commands.
modal_train_command.params = train_command.params
modal_evaluate_command.params = evaluate_command.params

cli.add_command(modal_train_command, name="train")
cli.add_command(modal_evaluate_command, name="evaluate")


if __name__ == "__main__":
    cli()
