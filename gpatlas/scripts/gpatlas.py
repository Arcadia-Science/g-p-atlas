import json
import pathlib
from typing import Literal

import click
import numpy as np
import torch

from gpatlas.datasets import GPAtlasDataset
from gpatlas.models import GPAtlasModel, GPAtlasModelConfig
from gpatlas.trainers import GPAtlasTrainer, GPAtlasTrainerConfig
from gpatlas.utils import compare_true_and_predicted_phenotypes


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "--dataset-filepath",
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
    "--hot-start-checkpoint-filepath",
    type=click.Path(path_type=pathlib.Path),
    required=False,
    help="Optional path to the checkpoint file from which to hot-start the training process.",
)
def train(
    dataset_filepath: pathlib.Path,
    output_dirpath: pathlib.Path,
    hot_start_checkpoint_filepath: pathlib.Path | None,
):
    """
    Main function for training a G-P Atlas model.

    The output directory must exist and must contain
    the `model_config.json` and `trainer_config.json` files.
    """

    # Create the output directory structure.
    output_dirpath.mkdir(parents=True, exist_ok=False)
    for subdir in ["checkpoints", "plots"]:
        (output_dirpath / subdir).mkdir(parents=True, exist_ok=False)

    with open(output_dirpath / "model_config.json") as file:
        model_config = GPAtlasModelConfig(**json.load(file))

    with open(output_dirpath / "trainer_config.json") as file:
        trainer_config = GPAtlasTrainerConfig(**json.load(file))

    train_dataset = GPAtlasDataset(
        filepath=str(dataset_filepath / "train.pk"),
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
    trainer.train(loader=train_loader)
    trainer.plot_training_losses(output_dirpath=output_dirpath)

    # TODO: save checkpoints at regular intervals during training.
    model.save_to_checkpoint(
        checkpoint_filepath=output_dirpath / "checkpoints" / "final_model_checkpoint.pt"
    )


@click.command()
@click.option(
    "--dataset-filepath",
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
def evaluate(dataset_filepath: pathlib.Path, output_dirpath: pathlib.Path):
    """
    Main function for evaluating a trained G-P Atlas model, using the final model checkpoint
    saved during training.
    """

    with open(output_dirpath / "model_config.json") as file:
        model_config = GPAtlasModelConfig(**json.load(file))

    with open(output_dirpath / "trainer_config.json") as file:
        trainer_config = GPAtlasTrainerConfig(**json.load(file))

    model = GPAtlasModel(config=model_config)
    model.load_from_checkpoint(
        checkpoint_filepath=output_dirpath / "checkpoints" / "final_model_checkpoint.pt"
    )

    trainer = GPAtlasTrainer(model=model, config=trainer_config)

    test_dataset = GPAtlasDataset(
        filepath=str(dataset_filepath / "test.pk"),
        num_phenotypes=model_config.num_input_phenotypes,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=1,
        num_workers=trainer_config.num_cores,
        # Important: we do *not* shuffle the test set so that the order of the samples
        # is preserved across evaluation loops.
        shuffle=False,
    )

    encoder_types: list[Literal["genotype", "phenotype"]] = ["genotype", "phenotype"]

    for encoder_type in encoder_types:
        true_phenotypes, decoded_phenotypes, embeddings = trainer.predict_phenotypes(
            loader=test_loader, encoder_type=encoder_type
        )

        feature_attributions = trainer.calculate_feature_importances(
            loader=test_loader, encoder_type=encoder_type
        )

        for array, filename in zip(
            [true_phenotypes, decoded_phenotypes, embeddings, feature_attributions],
            [
                "true_phenotypes.npy",
                "decoded_phenotypes.npy",
                "embeddings.npy",
                "feature_attributions.npy",
            ],
        ):
            np.save(output_dirpath / "predictions" / filename, array)

        compare_true_and_predicted_phenotypes(
            true_phenotypes=true_phenotypes,
            decoded_phenotypes=decoded_phenotypes,
            output_dirpath=(output_dirpath / "plots"),
            suffix=("PE" if encoder_type == "phenotype" else "GE"),
        )


if __name__ == "__main__":
    cli()
