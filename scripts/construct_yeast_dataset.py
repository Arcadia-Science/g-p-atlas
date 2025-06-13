import gzip
import pathlib
import pickle

import click
import h5py
import numpy as np


def construct_yeast_dataset(
    original_dataset_path: pathlib.Path, output_dirpath: pathlib.Path, train_test_split: float
) -> None:
    """
    Construct training and test sets for the yeast dataset from the `2025-g-p-atlas` repo.

    original_dataset_path: local path to the original dataset from the `2025-g-p-atlas` repo.
        This should be a pickled dictionary with keys "genotypes" and "phenotypes".
        Its location in the `2025-g-p-atlas` repo is:
            "data/yeast_data/g_p_atlas_format_data/full_yeast_data_torch_format.pk.gz"
    output_dirpath: local path to the directory where the constructed dataset will be stored.
    train_test_split: fraction of the dataset to use for training.
    """
    if original_dataset_path.suffix == ".gz":
        with gzip.open(original_dataset_path, "rb") as f:
            data = pickle.load(f)
    else:
        with open(original_dataset_path, "rb") as f:
            data = pickle.load(f)

    output_dirpath.mkdir(parents=True, exist_ok=True)

    # Genotypes are one-hot encoded, so we can use int8 to reduce file size.
    genotypes = np.array(data["genotypes"]).astype(np.int8)
    phenotypes = np.array(data["phenotypes"])

    if genotypes.shape[0] != phenotypes.shape[0]:
        raise ValueError("Genotypes and phenotypes must have the same number of samples.")

    inds = np.arange(phenotypes.shape[0])
    np.random.shuffle(inds)

    num_training_samples = np.floor(phenotypes.shape[0] * train_test_split).astype(np.int32)
    train_inds = inds[:num_training_samples]
    test_inds = inds[num_training_samples:]

    for split, inds in [("train", train_inds), ("test", test_inds)]:
        with h5py.File(output_dirpath / f"{split}-data.h5", "w") as file:
            file.create_dataset("genotypes", data=genotypes[inds])
            file.create_dataset("phenotypes", data=phenotypes[inds])

    click.echo(f"Constructed dataset saved to: '{output_dirpath}'")


@click.command()
@click.argument("original_dataset_path", type=click.Path(exists=True, path_type=pathlib.Path))
@click.argument("output_dataset_path", type=click.Path(path_type=pathlib.Path))
@click.option("--train-test-split", type=float, default=0.8)
def cli(
    original_dataset_path: pathlib.Path, output_dataset_path: pathlib.Path, train_test_split: float
) -> None:
    construct_yeast_dataset(original_dataset_path, output_dataset_path, train_test_split)


if __name__ == "__main__":
    cli()
