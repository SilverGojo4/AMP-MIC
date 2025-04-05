# pylint: disable=line-too-long, import-error, wrong-import-position, too-many-locals, too-many-statements, invalid-name, too-many-arguments, too-many-positional-arguments
"""
Machine Learning Utility Module

This module provides utility functions for the AMP-MIC project's machine learning components. It includes
helper functions for data preprocessing, configuration management, and logging support used across
model training, testing, and evaluation stages.

Core functions in this module:
- `preprocess_features()`: Standardizes feature values using StandardScaler.
- `extract_features_and_target()`: Extracts metadata, features, and target variables from a dataset.
- `read_json_config()`: Reads and parses a JSON configuration file.
- `get_hyperparameter_settings()`: Retrieves hyperparameter settings for a specific model type.
"""
# ============================== Standard Library Imports ==============================
import json
import logging
import os
import sys
from typing import Dict, Tuple

# ============================== Third-Party Library Imports ==============================
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ============================== Project Root Path Setup ==============================
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
LOGGING_PATH = os.path.join(BASE_PATH, "src/utils/logging_toolkit/src/python")

# Fix Python Path
if BASE_PATH not in sys.path:
    sys.path.append(BASE_PATH)
if LOGGING_PATH not in sys.path:
    sys.path.append(LOGGING_PATH)

# ============================== Project-Specific Imports ==============================
# Logging configuration and custom logger
from setup_logging import CustomLogger


# ============================== Custom Function ==============================
def read_json_config(config_path: str, logger: CustomLogger) -> Dict:
    """
    Read and parse a JSON configuration file.

    Parameters
    ----------
    config_path : str
        Path to the JSON file.
    logger : CustomLogger
        Logger instance.

    Returns
    -------
    dict
        Parsed JSON content.
    """
    try:
        # Load JSON file
        with open(config_path, "r", encoding="utf-8") as file:
            config = json.load(file)

        # Logging: Config Loaded
        logger.log_with_borders(
            level=logging.INFO,
            message=f"Loading hyperparameter settings:\n'{config_path}'",
            border="|",
            length=100,
        )
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

        return config

    except FileNotFoundError:
        logger.exception(msg="FileNotFoundError in 'read_json_config()'.")
        raise

    except json.JSONDecodeError:
        logger.exception(msg="JSONDecodeError in 'read_json_config()'.")
        raise

    except Exception:
        logger.exception(msg="Unexpected error in 'read_json_config()'.")
        raise


def get_hyperparameter_settings(
    config: Dict, model_type: str, logger: CustomLogger
) -> Dict:
    """
    Retrieve settings for a specific plot type from the config dictionary.

    Parameters
    ----------
    config : dict
        JSON-loaded dictionary containing plot settings.
    model_type : str
        The type of plot (e.g., 'BoxPlot', 'DistributionPlot').
    logger : CustomLogger
        Logger instance.

    Returns
    -------
    dict
        Settings for the specified plot type.
    """
    try:
        # Ensure the plot type exists in the config
        if model_type not in config:
            raise KeyError(f"Model type '{model_type}' not found in config.")

        # Logging: Retrieval Start
        logger.log_with_borders(
            level=logging.INFO,
            message=f"Successfully loaded hyperparameter configuration for '{model_type}'.",
            border="|",
            length=100,
        )

        return config[model_type]

    except Exception:
        logger.exception(msg="Unexpected error in 'get_hyperparameter_settings()'.")
        raise


def preprocess_features(X: pd.DataFrame, logger: CustomLogger) -> pd.DataFrame:
    """
    Standardize feature values using StandardScaler.

    Parameters
    ----------
    X : pd.DataFrame
        Feature dataframe containing numerical features.
    logger : CustomLogger
        Logger instance for tracking progress and errors.

    Returns
    -------
    pd.DataFrame
        Standardized feature dataframe with the same column names as the original.
    """
    try:
        logger.log_with_borders(
            level=logging.INFO,
            message="Starting 'feature scaling' using StandardScaler.",
            border="|",
            length=100,
        )
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
        logger.log_with_borders(
            level=logging.INFO,
            message=f"Feature scaling completed: {X_scaled_df.shape}.",
            border="|",
            length=100,
        )
        return X_scaled_df

    except Exception:
        logger.exception(msg="Unexpected error in 'preprocess_features()'.")
        raise


def extract_features_and_target(
    file_path: str,
    metadata_columns: list,
    target_column: str,
    feature_start_idx: int,
    feature_end_idx: int,
    logger: CustomLogger,
    standardize: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Extract metadata, features, and target by index range.

    Parameters
    ----------
    file_path : str
        Path to the CSV file.
    metadata_columns : list
        List of metadata columns (e.g., ['ID', 'Sequence', 'Targets']).
    target_column : str
        Target variable column name.
    feature_start_idx : int
        Start index of feature columns.
    feature_end_idx : int
        End index of feature columns.
    logger : CustomLogger
        Logger instance for tracking progress and errors.
    standardize : bool, optional
        Whether to standardize the features using StandardScaler. Default is True.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series]
        Metadata, scaled features, and target variable.
    """
    try:

        # Log the start of data extraction
        logger.info(msg="/ Task: Extracting features for MIC prediction")
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")
        logger.log_with_borders(
            level=logging.INFO,
            message=f"Loading data from '{file_path}'.",
            border="|",
            length=100,
        )
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

        # Read CSV file into a DataFrame
        df = pd.read_csv(file_path)
        if df.empty:
            raise ValueError(f"'{file_path}' is empty.")
        if target_column not in df.columns:
            raise KeyError(f"Missing '{target_column}'.")

        # Extract metadata (e.g., ID, Sequence, Targets)
        df_metadata = df[metadata_columns]

        # Select feature columns using the specified index range
        feature_columns = df.columns[feature_start_idx : feature_end_idx + 1]
        X = df[feature_columns]
        X = X.apply(pd.to_numeric, errors="coerce")
        y = df[target_column]

        # Ensure the target column exists and contains no missing values
        if y.isnull().any():
            raise ValueError(f"'{target_column}' contains missing values.")

        # Retrieve the names of the first and last feature columns
        first_feature = feature_columns[0] if len(feature_columns) > 0 else "None"
        last_feature = feature_columns[-1] if len(feature_columns) > 0 else "None"

        # Log the number of features and target values extracted
        logger.log_with_borders(
            level=logging.INFO,
            message=(
                f"Extracted {X.shape[1]} features and {len(y)} targets.\n"
                f"First feature: '{first_feature}'\n"
                f"Last feature: '{last_feature}'"
            ),
            border="|",
            length=100,
        )
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

        # Apply standardization if specified
        if standardize:
            X_processed = preprocess_features(X, logger)
        else:
            logger.log_with_borders(
                level=logging.INFO,
                message="Skipping feature standardization as per request.\n"
                f"Feature completed: {X.shape}.",
                border="|",
                length=100,
            )
            X_processed = X

        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

        return df_metadata, X_processed, y

    except ValueError:
        logger.exception(msg="ValueError in 'extract_features_and_target()'.")
        raise
    except KeyError:
        logger.exception(msg="KeyError in 'extract_features_and_target()'.")
        raise

    except Exception:
        logger.exception(msg="Unexpected error in 'extract_features_and_target()'.")
        raise
