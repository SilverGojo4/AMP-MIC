# pylint: disable=line-too-long, import-error, wrong-import-position, too-many-locals, too-many-statements, invalid-name, too-many-arguments, too-many-positional-arguments
"""
Deep Learning Utility Module

This module provides utility functions for the AMP-MIC project's deep learning components. It includes
helper functions for data preprocessing, configuration management, and logging support used across
model training, testing, and evaluation stages.

Core functions in this module:
- `extract_features_and_target_for_dl()`: Extracts metadata, features, and target variables from a dataset for deep learning.
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
import numpy as np
import pandas as pd
import torch

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


def extract_cgr_features_and_target_for_dl(
    file_path: str,
    metadata_columns: list,
    target_column: str,
    feature_start_idx: int,
    feature_end_idx: int,
    height: int,
    width: int,
    logger: CustomLogger,
) -> Tuple[pd.DataFrame, torch.Tensor, torch.Tensor]:
    """
    Extract metadata, features, and target by index range for deep learning.

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
    height : int
        Height of the feature map (e.g., 16 for 16x16).
    width : int
        Width of the feature map (e.g., 16 for 16x16).
    logger : CustomLogger
        Logger instance for tracking progress and errors.

    Returns
    -------
    Tuple[pd.DataFrame, torch.Tensor, torch.Tensor]
        Metadata, features as a tensor of shape (batch_size, in_channels, height, width),
        and target variable as a tensor of shape (batch_size, 1).
    """
    try:
        # Log the start of data extraction
        logger.info(
            msg="/ Task: Extracting features for MIC prediction ('Deep Learning')"
        )
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

        # Check feature size
        expected_size = height * width
        if X.shape[1] != expected_size:
            raise ValueError(
                f"Feature size ({X.shape[1]}) does not match expected size ({expected_size}) for {height}x{width} feature map."
            )

        # Convert to torch tensors
        X_tensor = torch.tensor(
            X.values, dtype=torch.float32
        )  # (batch_size, height*width)
        y_tensor = torch.tensor(y.values, dtype=torch.float32).view(
            -1, 1
        )  # (batch_size, 1)

        # Reshape features to (batch_size, in_channels, height, width)
        X_tensor = X_tensor.view(-1, 1, height, width)  # (batch_size, 1, height, width)

        # Log the number of features and target values extracted
        logger.log_with_borders(
            level=logging.INFO,
            message=(
                f"Feature shape after reshape: {X_tensor.shape}\n"
                f"Target shape: {y_tensor.shape}"
            ),
            border="|",
            length=100,
        )
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

        return df_metadata, X_tensor, y_tensor

    except ValueError:
        logger.exception(
            msg="ValueError in 'extract_cgr_features_and_target_for_dl()'."
        )
        raise

    except KeyError:
        logger.exception(msg="KeyError in 'extract_cgr_features_and_target_for_dl()'.")
        raise

    except Exception:
        logger.exception(
            msg="Unexpected error in 'extract_cgr_features_and_target_for_dl()'."
        )
        raise


def extract_word_embedding_features_for_dl(
    npz_path: str,
    df_path: str,
    metadata_columns: list,
    target_column: str,
    height: int,
    width: int,
    logger: CustomLogger,
) -> Tuple[pd.DataFrame, torch.Tensor, torch.Tensor]:
    """
    Extract metadata, word embedding features, and target for deep learning from .npz and CSV files.

    This function processes 1D word embedding features with shape (batch_size, in_channels, max_len),
    suitable for models like WordEmbeddingANIA. It does not reshape data into 2D format.

    Parameters
    ----------
    npz_path : str
        Path to the .npz file containing word embedding data.
    df_path : str
        Path to the CSV file containing metadata and target data.
    metadata_columns : list
        List of metadata columns (e.g., ['ID', 'Sequence', 'Targets']).
    target_column : str
        Target variable column name (e.g., 'Log MIC Value').
    height : int
        Expected sequence length (max_len) of the word embedding features.
    width : Optional[int]
        Not used for 1D word embedding data (should be None).
    logger : CustomLogger
        Logger instance for tracking progress and errors.

    Returns
    -------
    Tuple[pd.DataFrame, torch.Tensor, torch.Tensor]
        Metadata DataFrame, features as a tensor of shape (batch_size, in_channels, max_len),
        and target variable as a tensor of shape (batch_size, 1).
    """
    try:
        logger.info(
            msg="/ Task: Extracting word embedding features for MIC prediction ('Deep Learning')"
        )
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")
        logger.log_with_borders(
            level=logging.INFO,
            message=f"Loading embedding data from '{npz_path}' and metadata from '{df_path}'.",
            border="|",
            length=100,
        )
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

        # Load .npz file
        data = np.load(npz_path, allow_pickle=True)
        identifiers = data["identifiers"]  # Sequence IDs from FASTA
        encoded_matrices = data[
            "encoded_matrices"
        ]  # (num_sequences, max_len, embedding_dim)
        max_len = data["max_len"].item()
        _ = data["embedding_dim"].item()

        # Load CSV file for metadata and target
        df = pd.read_csv(df_path)
        if df.empty:
            raise ValueError(f"'{df_path}' is empty.")
        if target_column not in df.columns:
            raise KeyError(f"Missing '{target_column}' in CSV.")

        # Extract metadata and target
        df_metadata = df[metadata_columns]
        y = df[target_column]
        if y.isnull().any():
            raise ValueError(f"'{target_column}' contains missing values.")

        # Ensure IDs match between .npz and CSV
        csv_ids = df["ID"].values
        if not np.array_equal(identifiers, csv_ids):  # type: ignore
            logger.warning(
                "Sequence IDs in .npz and CSV do not fully match. Aligning by ID."
            )
            df = df.set_index("ID").reindex(identifiers).reset_index()
            df_metadata = df[metadata_columns]
            y = df[target_column]

        # Verify feature length
        if max_len != height:
            raise ValueError(
                f"Feature length mismatch: expected max_len={height}, got {max_len}."
            )
        if width is not None:
            logger.warning(
                "Width parameter is ignored for 1D word embedding data (expected None)."
            )

        # Shape remains (batch_size, max_len, embedding_dim), then permute to (batch_size, embedding_dim, max_len)
        X_tensor = torch.tensor(
            encoded_matrices, dtype=torch.float32
        )  # (batch_size, max_len, embedding_dim)
        X_tensor = X_tensor.permute(0, 2, 1)  # (batch_size, embedding_dim, max_len)
        y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)

        logger.log_with_borders(
            level=logging.INFO,
            message=f"Feature shape after reshape: {X_tensor.shape}\n"
            f"Target shape: {y_tensor.shape}",
            border="|",
            length=100,
        )
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

        return df_metadata, X_tensor, y_tensor

    except Exception:
        logger.exception(
            msg="Unexpected error in 'extract_word_embedding_features_for_dl()'."
        )
        raise
