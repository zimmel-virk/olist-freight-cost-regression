"""Reusable utilities extracted from the CM3005 freight-cost project.

The functions here are reorganised from the original notebook so the core
data-quality, geographical and evaluation logic can be reused independently.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)


def sanitise_string_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from text columns and convert empty strings to missing."""
    sanitised_dataframe = dataframe.copy(deep=True)

    string_columns = sanitised_dataframe.select_dtypes(
        include=["object", "string"]
    ).columns

    for column_name in string_columns:
        sanitised_dataframe[column_name] = (
            sanitised_dataframe[column_name]
            .astype("string")
            .str.strip()
            .replace(r"^\s*$", pd.NA, regex=True)
        )

    return sanitised_dataframe


def convert_datetime_columns(
    dataframe: pd.DataFrame,
    dataset_name: str,
    datetime_columns: list[str],
):
    """Convert selected fields to datetime and return an audit trail."""
    converted_dataframe = dataframe.copy()
    conversion_records = []

    for column_name in datetime_columns:
        original_non_missing = int(
            converted_dataframe[column_name].notna().sum()
        )

        converted_values = pd.to_datetime(
            converted_dataframe[column_name],
            format="%Y-%m-%d %H:%M:%S",
            errors="coerce",
        )

        converted_non_missing = int(converted_values.notna().sum())
        parsing_failures = original_non_missing - converted_non_missing
        converted_dataframe[column_name] = converted_values

        conversion_records.append(
            {
                "dataset": dataset_name,
                "column": column_name,
                "original_non_missing": original_non_missing,
                "converted_non_missing": converted_non_missing,
                "parsing_failures": parsing_failures,
                "final_data_type": str(converted_dataframe[column_name].dtype),
            }
        )

    return converted_dataframe, conversion_records


def convert_numeric_columns(
    dataframe: pd.DataFrame,
    dataset_name: str,
    numerical_columns: list[str],
):
    """Convert selected fields to numeric data and return conversion failures."""
    converted_dataframe = dataframe.copy()
    conversion_records = []

    for column_name in numerical_columns:
        original_non_missing = int(
            converted_dataframe[column_name].notna().sum()
        )

        converted_values = pd.to_numeric(
            converted_dataframe[column_name],
            errors="coerce",
        )

        converted_non_missing = int(converted_values.notna().sum())
        conversion_failures = original_non_missing - converted_non_missing
        converted_dataframe[column_name] = converted_values

        conversion_records.append(
            {
                "dataset": dataset_name,
                "column": column_name,
                "original_non_missing": original_non_missing,
                "converted_non_missing": converted_non_missing,
                "conversion_failures": conversion_failures,
                "final_data_type": str(converted_dataframe[column_name].dtype),
            }
        )

    return converted_dataframe, conversion_records


def calculate_haversine_distance(
    latitude_1,
    longitude_1,
    latitude_2,
    longitude_2,
):
    """Great-circle distance in kilometres between coordinate pairs."""
    earth_radius_km = 6_371.0088

    latitude_1_radians = np.radians(latitude_1)
    longitude_1_radians = np.radians(longitude_1)
    latitude_2_radians = np.radians(latitude_2)
    longitude_2_radians = np.radians(longitude_2)

    latitude_difference = latitude_2_radians - latitude_1_radians
    longitude_difference = longitude_2_radians - longitude_1_radians

    haversine_component = (
        np.sin(latitude_difference / 2) ** 2
        + np.cos(latitude_1_radians)
        * np.cos(latitude_2_radians)
        * np.sin(longitude_difference / 2) ** 2
    )

    haversine_component = np.clip(haversine_component, 0, 1)
    angular_distance = 2 * np.arcsin(np.sqrt(haversine_component))

    return earth_radius_km * angular_distance


def calculate_rmse(actual_values, predicted_values) -> float:
    return float(
        np.sqrt(mean_squared_error(actual_values, predicted_values))
    )


def calculate_regression_metrics(
    actual_values,
    predicted_values,
    model_name: str,
):
    """Return the project's principal regression evaluation measures."""
    rmse_value = calculate_rmse(actual_values, predicted_values)
    mae_value = mean_absolute_error(actual_values, predicted_values)
    median_ae_value = median_absolute_error(actual_values, predicted_values)
    r_squared_value = r2_score(actual_values, predicted_values)
    target_mean = np.mean(actual_values)

    return {
        "model": model_name,
        "RMSE": rmse_value,
        "MAE": float(mae_value),
        "median_absolute_error": float(median_ae_value),
        "R_squared": float(r_squared_value),
        "RMSE_as_percentage_of_target_mean": (
            float(rmse_value / target_mean * 100)
            if target_mean != 0
            else np.nan
        ),
        "predicted_negative_values": int(
            np.sum(np.asarray(predicted_values) < 0)
        ),
    }


def bootstrap_confidence_interval(values, confidence_level=0.95):
    """Return the bootstrap mean and percentile confidence interval."""
    lower_percentile = (1 - confidence_level) / 2 * 100
    upper_percentile = 100 - lower_percentile

    values = pd.Series(values).dropna()

    return {
        "mean": float(values.mean()),
        "lower_confidence_limit": float(
            np.percentile(values, lower_percentile)
        ),
        "upper_confidence_limit": float(
            np.percentile(values, upper_percentile)
        ),
    }
