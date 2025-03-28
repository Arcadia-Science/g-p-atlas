import json
import os
import pathlib

import numpy as np
import pytest

from gpatlas.utils import compare_true_and_predicted_phenotypes, mean_absolute_percentage_error


@pytest.mark.parametrize(
    "y_true, y_pred, expected",
    [
        (
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
            0.0,
        ),
        (
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
            np.array([2.0, 4.0, 6.0], dtype=np.float32),
            100.0,
        ),
        (
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
            np.array([0.5, 1.0, 1.5], dtype=np.float32),
            50.0,
        ),
    ],
)
def test_mean_absolute_percentage_error(y_true, y_pred, expected):
    """
    Test the MAPE calculation function.
    """
    result = mean_absolute_percentage_error(y_true, y_pred)
    assert np.isclose(result, expected)


def test_mean_absolute_percentage_error_with_zeros():
    """
    Test that the MAPE calculation handles zeros in y_true.
    """
    y_true = np.array([0.0, 2.0, 3.0], dtype=np.float32)
    y_pred = np.array([0.1, 1.8, 3.3], dtype=np.float32)

    result = mean_absolute_percentage_error(y_true, y_pred)

    # Check that the result is finite.
    assert np.isfinite(result)


def test_compare_true_and_predicted_phenotypes(tmp_path: pathlib.Path):
    """
    Test the function that compares true and predicted phenotypes.
    """
    n_samples = 20
    n_phenotypes = 5

    true_phenotypes = np.random.rand(n_samples, n_phenotypes).astype(np.float32)
    decoded_phenotypes = true_phenotypes + 0.1 * np.random.randn(n_samples, n_phenotypes).astype(
        np.float32
    )

    suffix = "some-suffix"
    output_dirpath = tmp_path / "test"
    output_dirpath.mkdir(parents=True, exist_ok=True)

    compare_true_and_predicted_phenotypes(
        true_phenotypes=true_phenotypes,
        decoded_phenotypes=decoded_phenotypes,
        output_dirpath=output_dirpath,
        suffix=suffix,
    )

    # Check that the expected files were created.
    expected_files = [
        f"true_vs_pred_phenotypes_{suffix}.svg",
        f"true_vs_pred_phenotypes_pearson_correlation_{suffix}.svg",
        f"true_vs_pred_phenotypes_mse_{suffix}.svg",
        f"true_vs_pred_phenotypes_mape_{suffix}.svg",
        f"true_vs_pred_phenotypes_r2_{suffix}.svg",
        f"true_vs_pred_phenotypes_summary_stats_{suffix}.json",
    ]
    for filename in expected_files:
        assert os.path.exists(output_dirpath / filename)

    # Check that the JSON file contains the expected keys
    with open(output_dirpath / f"true_vs_pred_phenotypes_summary_stats_{suffix}.json") as f:
        stats = json.load(f)

    expected_keys = ["pearson_correlation", "mse", "mape", "r2"]
    for key in expected_keys:
        assert key in stats
        assert len(stats[key]) == n_phenotypes
