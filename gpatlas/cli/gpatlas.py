import pathlib
from typing import Literal

import click
import numpy as np
import torch
import yaml

from gpatlas.datasets import GPAtlasDataset
from gpatlas.models import GPAtlasModel, GPAtlasModelConfig
from gpatlas.trainers import GPAtlasTrainer, GPAtlasTrainerConfig
from gpatlas.utils import compare_true_and_predicted_phenotypes


def load_configs(output_dirpath: pathlib.Path) -> tuple[GPAtlasModelConfig, GPAtlasTrainerConfig]:
    """
    Load the model and trainer configuration files from the output directory.
    """
    model_config_filepath = output_dirpath / "model-config.yaml"
    trainer_config_filepath = output_dirpath / "trainer-config.yaml"

    if not model_config_filepath.exists():
        raise FileNotFoundError(f"Model configuration file not found at {model_config_filepath}")

    if not trainer_config_filepath.exists():
        raise FileNotFoundError(
            f"Trainer configuration file not found at {trainer_config_filepath}"
        )

    with open(model_config_filepath) as file:
        model_config = GPAtlasModelConfig(**(yaml.safe_load(file) or {}))

    with open(trainer_config_filepath) as file:
        trainer_config = GPAtlasTrainerConfig(**(yaml.safe_load(file) or {}))

    return model_config, trainer_config


def dump_configs(
    model_config: GPAtlasModelConfig,
    trainer_config: GPAtlasTrainerConfig,
    output_dirpath: pathlib.Path,
) -> None:
    """
    Dump the model and trainer configuration files to the output directory.
    """
    with open(output_dirpath / "model-config.yaml", "w") as file:
        yaml.safe_dump(model_config.model_dump(), file)

    with open(output_dirpath / "trainer-config.yaml", "w") as file:
        yaml.safe_dump(trainer_config.model_dump(), file)


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "--dataset-dirpath",
    type=click.Path(exists=True, path_type=pathlib.Path),
    required=True,
    help="Path to the directory containing the training and test dataset files.",
)
@click.option(
    "--output-dirpath",
    type=click.Path(exists=True, path_type=pathlib.Path),
    required=True,
    help="Path to the directory where all output files should be saved.",
)
@click.option(
    "--hot-start-checkpoint-filepath",
    type=click.Path(exists=True, path_type=pathlib.Path),
    required=False,
    help="Optional path to the checkpoint file from which to hot-start the training process.",
)
@click.option(
    "--detect-anomaly",
    is_flag=True,
    help="Enable anomaly detection.",
)
def train(
    dataset_dirpath: pathlib.Path,
    output_dirpath: pathlib.Path,
    hot_start_checkpoint_filepath: pathlib.Path | None,
    detect_anomaly: bool,
):
    """
    Train a G-P Atlas model.

    Note: the output directory must exist and must contain valid configuration files
    (`model_config.yaml` and `trainer_config.yaml`) that define the model and trainer
    configurations, respectively.
    """

    # Create the output directory structure.
    for subdir in ["checkpoints", "plots", "predictions", "logs"]:
        (output_dirpath / subdir).mkdir(parents=True, exist_ok=True)

    model_config, trainer_config = load_configs(output_dirpath)

    # Save the configs to capture the actual values.
    dump_configs(model_config, trainer_config, output_dirpath / "logs")

    train_dataset = GPAtlasDataset(
        filepath=str(dataset_dirpath / "train-data.h5"),
        num_phenotypes=model_config.num_input_phenotypes,
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=trainer_config.batch_size,
        num_workers=trainer_config.num_cores,
        shuffle=True,
    )

    model = GPAtlasModel(config=model_config)

    if hot_start_checkpoint_filepath is not None:
        model.load_from_checkpoint(hot_start_checkpoint_filepath)

    trainer = GPAtlasTrainer(model=model, config=trainer_config)

    if detect_anomaly:
        with torch.autograd.set_detect_anomaly(True, check_nan=True):
            trainer.train(loader=train_loader)
    else:
        trainer.train(loader=train_loader)

    trainer.plot_training_losses(filepath=output_dirpath / "plots" / "training_losses.svg")

    # TODO: save checkpoints at regular intervals during training.
    model.save_to_checkpoint(
        checkpoint_filepath=output_dirpath / "checkpoints" / "final_model_checkpoint.pt"
    )


@cli.command()
@click.option(
    "--dataset-dirpath",
    type=click.Path(exists=True, path_type=pathlib.Path),
    required=True,
    help="Path to the directory containing the training and test dataset files.",
)
@click.option(
    "--output-dirpath",
    type=click.Path(path_type=pathlib.Path),
    required=True,
    help="Path to the directory where all output files should be saved.",
)
@click.option(
    "--calculate-feature-importances",
    is_flag=True,
    help="Calculate feature importances for the genotype and phenotype encoders.",
)
def evaluate(
    dataset_dirpath: pathlib.Path, output_dirpath: pathlib.Path, calculate_feature_importances: bool
):
    """
    Evaluate a trained G-P Atlas model.

    Note: the evaluation uses the final model checkpoint saved during training.
    """
    checkpoint_filepath = output_dirpath / "checkpoints" / "final_model_checkpoint.pt"

    model_config, trainer_config = load_configs(output_dirpath)
    model = GPAtlasModel(config=model_config)

    model.load_from_checkpoint(checkpoint_filepath)
    print(f"Loaded model from {checkpoint_filepath}")

    trainer = GPAtlasTrainer(model=model, config=trainer_config)

    test_dataset = GPAtlasDataset(
        filepath=str(dataset_dirpath / "test-data.h5"),
        num_phenotypes=model_config.num_input_phenotypes,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        # TODO: why use a batch size of 1?
        batch_size=1,
        num_workers=trainer_config.num_cores,
        # Important: we do *not* shuffle the test set so that the order of the samples
        # is preserved across evaluation loops.
        shuffle=False,
    )

    input_types: list[Literal["genotype", "phenotype"]] = ["genotype", "phenotype"]

    # The suffixes 'ge' and 'pe' are abbreviations to distinguish between results
    # from using the genotypes or phenotypes, respectively, as the inputs to the model.
    filename_suffixes = ["ge", "pe"]

    for input_type, filename_suffix in zip(input_types, filename_suffixes):
        print(f"Predicting phenotypes for {len(test_loader)} samples...")

        true_phenotypes, decoded_phenotypes, embeddings = trainer.predict_phenotypes(
            loader=test_loader, input_type=input_type
        )
        for array, name in zip(
            [true_phenotypes, decoded_phenotypes, embeddings],
            ["true_phenotypes", "decoded_phenotypes", "embeddings"],
        ):
            np.save(output_dirpath / "predictions" / f"{name}_{filename_suffix}.npy", array)

        # Generate and save plots of the predicted vs true phenotypes.
        compare_true_and_predicted_phenotypes(
            true_phenotypes=true_phenotypes,
            decoded_phenotypes=decoded_phenotypes,
            output_dirpath=(output_dirpath / "plots"),
            suffix=filename_suffix,
        )

        if calculate_feature_importances:
            print(f"Calculating feature importances for {len(test_loader)} samples...")
            feature_attributions = trainer.calculate_feature_importances(
                loader=test_loader, input_type=input_type
            )
            feature_attributions_filepath = (
                output_dirpath / "predictions" / f"feature_attributions_{filename_suffix}.npy"
            )
            np.save(feature_attributions_filepath, feature_attributions)


if __name__ == "__main__":
    cli()
