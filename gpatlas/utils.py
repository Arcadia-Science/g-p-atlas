import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import scipy
import sklearn.metrics
import torch
from numpy.typing import NDArray

from gpatlas.constants import EPS


def get_device(device_name: str | None = None) -> torch.device:
    """
    Get the PyTorch device to use for training.
    """
    if device_name is None or device_name == "auto":
        device_name = (
            "mps"
            if torch.backends.mps.is_available()
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    return torch.device(device_name)


def mean_absolute_percentage_error(
    y_true: NDArray[np.float32], y_pred: NDArray[np.float32]
) -> float:
    """
    Calculate the Mean Absolute Percentage Error between true and predicted values.
    """
    y_true = np.array(y_true + EPS)
    y_pred = np.array(y_pred)
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def compare_true_and_predicted_phenotypes(
    true_phenotypes: NDArray[np.float32],
    decoded_phenotypes: NDArray[np.float32],
    output_dirpath: str | pathlib.Path,
    suffix: str = "",
) -> None:
    """
    Generate plots and summary statistics comparing the true and decoded phenotypes.

    Parameters
    ----------
    true_phenotypes : array of true phenotypes of shape `(num_samples, num_phenotypes)`.
    decoded_phenotypes : array of decoded phenotypes of shape `(num_samples, num_phenotypes)`.
    output_dirpath : Path to directory where all output files should be saved
    suffix : str, optional
        Suffix to add to the output file names.

    Returns
    -------
    None
    """
    output_dirpath = pathlib.Path(output_dirpath)

    num_phenotypes = true_phenotypes.shape[1]

    # Plot decoded phenotypes vs. true phenotypes.
    for ind in range(num_phenotypes):
        plt.plot(true_phenotypes[:, ind], decoded_phenotypes[:, ind], "o")

    plt.xlabel("True phenotype")
    plt.ylabel("Predicted phenotype")
    plt.gca().set_aspect("equal")
    plt.savefig(output_dirpath / f"true_vs_pred_phenotypes_{suffix}.svg")
    plt.close()

    summary_stats: dict[str, list[float]] = {}

    # Pearson correlation.
    pearson_correlations = [
        float(scipy.stats.pearsonr(true_phenotypes[:, ind], decoded_phenotypes[:, ind])[0])
        for ind in range(num_phenotypes)
    ]
    summary_stats["pearson_correlation"] = pearson_correlations
    plt.hist(pearson_correlations, bins=20)
    plt.savefig(output_dirpath / f"true_vs_pred_phenotypes_pearson_correlation_{suffix}.svg")
    plt.close()

    # Mean squared error.
    mse_values = sklearn.metrics.mean_squared_error(
        true_phenotypes, decoded_phenotypes, multioutput="raw_values"
    )
    summary_stats["mse"] = mse_values.astype(float).tolist()  # type: ignore
    plt.hist(mse_values, bins=20)
    plt.savefig(output_dirpath / f"true_vs_pred_phenotypes_mse_{suffix}.svg")
    plt.close()

    # Mean absolute percentage error.
    mape_values = [
        mean_absolute_percentage_error(true_phenotypes[:, ind], decoded_phenotypes[:, ind])
        for ind in range(num_phenotypes)
    ]
    summary_stats["mape"] = mape_values
    plt.hist(mape_values, bins=20)
    plt.savefig(output_dirpath / f"true_vs_pred_phenotypes_mape_{suffix}.svg")
    plt.close()

    # R-squared score (coefficient of determination).
    r2_values = sklearn.metrics.r2_score(
        true_phenotypes, decoded_phenotypes, multioutput="raw_values"
    )
    summary_stats["r2"] = r2_values.astype(float).tolist()  # type: ignore
    plt.hist(r2_values, bins=20)
    plt.savefig(output_dirpath / f"true_vs_pred_phenotypes_r2_{suffix}.svg")
    plt.close()

    with open(output_dirpath / f"true_vs_pred_phenotypes_summary_stats_{suffix}.json", "w") as file:
        json.dump(summary_stats, file)
