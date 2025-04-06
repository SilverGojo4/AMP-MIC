# pylint: disable=line-too-long, import-error, wrong-import-position, too-many-locals, too-many-statements, invalid-name, too-many-arguments, too-many-positional-arguments, too-many-branches, too-many-nested-blocks
"""
Deep Learning Model Testing Module

This module provides functionality for testing trained deep learning models (e.g., FCGRANIA, WordEmbeddingANIA, ANIA)
on AMP-MIC data to predict Minimum Inhibitory Concentration (MIC). It evaluates model performance
using standard regression metrics and supports structured logging.
"""
# ============================== Standard Library Imports ==============================
import logging
import os
import sys
import time
from typing import Optional

# ============================== Third-Party Library Imports ==============================
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, r2_score
from torch import nn

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

# Deep Learning Utility Functions
from src.models.deep_learning.ania import ANIA
from src.models.deep_learning.fcgr_ania import FCGRANIA
from src.models.deep_learning.utils import (
    extract_cgr_features_and_target_for_dl,
    extract_word_embedding_features_for_dl,
)
from src.models.deep_learning.word_embedding_ania import WordEmbeddingANIA


# ============================== Custom Function ==============================
def evaluate_dl_predictions(
    predictions_csv_path: str,
    model_type: str,
    logger: CustomLogger,
) -> None:
    """
    Compute evaluation metrics from saved prediction CSV.

    Parameters
    ----------
    predictions_csv_path : str
        Path to the CSV file containing actual and predicted values.
    model_type : str
        Type of model to train ('fcgr_ania').
    logger : CustomLogger
        Logger instance for tracking progress and errors.

    Returns
    -------
    None
    """
    try:
        # Load predictions
        df = pd.read_csv(predictions_csv_path)

        # Ensure required columns exist
        if (
            "Log MIC Value" not in df.columns
            or "Predicted Log MIC Value" not in df.columns
        ):
            raise KeyError("Missing required columns in prediction CSV.")

        # Extract actual and predicted values
        y_true = df["Log MIC Value"]
        y_pred = df["Predicted Log MIC Value"]

        # Compute evaluation metrics
        mae = np.mean(np.abs(y_true - y_pred))
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        pcc = np.corrcoef(y_true, y_pred)[0, 1]  # Pearson Correlation Coefficient

        # Log results
        logger.log_with_borders(
            level=logging.INFO,
            message=(
                f"Evaluation Metrics for '{model_type}' model:\n"
                f"  • Mean Absolute Error (MAE): {mae:.4f}\n"
                f"  • Mean Squared Error (MSE): {mse:.4f}\n"
                f"  • Root Mean Squared Error (RMSE): {rmse:.4f}\n"
                f"  • R² Score: {r2:.4f}\n"
                f"  • Pearson Correlation Coefficient (PCC): {pcc:.4f}"
            ),
            border="|",
            length=100,
        )

    except FileNotFoundError:
        logger.exception(msg="FileNotFoundError in 'evaluate_dl_predictions()'.")
        raise

    except KeyError:
        logger.exception(msg="KeyError in 'evaluate_dl_predictions()'.")
        raise

    except Exception:
        logger.exception(msg="Unexpected error in 'evaluate_dl_predictions()'.")
        raise


def load_trained_model(
    model_type: str,
    checkpoint_path: str,
    device: str,
    logger: CustomLogger,
) -> nn.Module:
    """
    Load a trained deep learning model from checkpoint.

    Parameters
    ----------
    model_type : str
        Type of model to load ('fcgr_ania', 'word_embedding_ania', or 'ania').
    checkpoint_path : str
        Path to the model checkpoint file.
    device : str
        Device to load the model onto (e.g., 'cuda:0').
    logger : CustomLogger
        Logger instance for tracking progress and errors.

    Returns
    -------
    nn.Module
        Loaded model instance ready for inference.
    """
    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Extract hyperparameters used during training
        hyperparams = checkpoint.get("hyperparams")
        if hyperparams is None:
            raise ValueError("Missing 'hyperparams' in checkpoint file.")

        # Initialize model based on model_type
        if model_type == "fcgr_ania":
            model = FCGRANIA(
                in_channels=1,
                inception_out_channels=hyperparams["inception_out_channels"],
                num_heads=hyperparams["num_heads"],
                d_model=hyperparams["d_model"],
                dense_hidden_dim=hyperparams["dense_hidden_dim"],
                dropout_rate=hyperparams["dropout_rate"],
            ).to(device)
        elif model_type == "word_embedding_ania":
            model = WordEmbeddingANIA(
                in_channels=checkpoint["input_shape"][
                    0
                ],  # embedding_dim from checkpoint
                inception_out_channels=hyperparams["inception_out_channels"],
                num_heads=hyperparams["num_heads"],
                d_model=hyperparams["d_model"],
                dense_hidden_dim=hyperparams["dense_hidden_dim"],
                dropout_rate=hyperparams["dropout_rate"],
            ).to(device)
        elif model_type == "ania":
            model = ANIA(
                fcgr_in_channels=1,
                word_in_channels=checkpoint["input_shape"]["word"][
                    0
                ],  # from dictionary
                fcgr_inception_out_channels=hyperparams["fcgr_inception_out_channels"],
                word_inception_out_channels=hyperparams["word_inception_out_channels"],
                num_heads=hyperparams["num_heads"],
                d_model=hyperparams["d_model"],
                dense_hidden_dim=hyperparams["dense_hidden_dim"],
                dropout_rate=hyperparams["dropout_rate"],
            ).to(device)
        else:
            raise ValueError(f"Unsupported model type: '{model_type}'")

        # Load weights
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        logger.log_with_borders(
            level=logging.INFO,
            message=f"Successfully loaded trained '{model_type}' model from checkpoint:\n'{checkpoint_path}'",
            border="|",
            length=100,
        )
        return model

    except Exception:
        logger.exception(
            msg=f"Unexpected error in 'load_trained_model()' for '{model_type}'."
        )
        raise


def test_dl_model(
    df_metadata: pd.DataFrame,
    X_test_fcgr: torch.Tensor,  # FCGR input for ANIA
    X_test_word: Optional[
        torch.Tensor
    ],  # Word embedding input for ANIA (None for single-input models)
    y_test: torch.Tensor,
    model_type: str,
    model_input_path: str,
    prediction_output_path: str,
    logger: CustomLogger,
    device: str = "cuda:0",
) -> None:
    """
    Test a trained deep learning model on test data and save predictions.

    Parameters
    ----------
    df_metadata : pd.DataFrame
        DataFrame containing metadata (e.g., ID, Sequence, Targets).
    X_test_fcgr : torch.Tensor
        FCGR test feature tensor of shape (batch_size, in_channels, height, width) for 'fcgr_ania' or 'ania'.
        For 'word_embedding_ania', this is the word embedding feature set if X_test_word is None.
    X_test_word : Optional[torch.Tensor]
        Word embedding test feature tensor of shape (batch_size, in_channels, max_len) for 'word_embedding_ania' or 'ania'.
        None for single-input models like 'fcgr_ania'.
    y_test : torch.Tensor
        Test target tensor of shape (batch_size, 1).
    model_type : str
        Type of deep learning model to test ('fcgr_ania', 'word_embedding_ania', or 'ania').
    model_input_path : str
        Path to the trained model checkpoint (.pt file).
    prediction_output_path : str
        Path to save the prediction results as CSV.
    logger : CustomLogger
        Logger instance for tracking progress and errors.
    device : str, optional
        Device to run the model on (e.g., 'cuda:0', default: 'cuda:0').

    Returns
    -------
    None
    """
    try:
        # Log training start
        logger.info(msg=f"/ Task: Test '{model_type}' model")
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

        # Configure the device (GPU or CPU) and log GPU information if applicable
        gpu_id = None
        if "cuda" in device:
            num_gpus = torch.cuda.device_count()
            device_idx = int(device.split(":")[1]) if ":" in device else 0
            if device_idx >= num_gpus:
                device = "cuda:0"
            if not torch.cuda.is_available():
                logger.log_with_borders(
                    level=logging.WARNING,
                    message="CUDA is not available, falling back to CPU.",
                    border="|",
                    length=100,
                )
                device = "cpu"
            else:
                torch.cuda.set_device(device)
                gpu_id = torch.cuda.current_device()
                if gpu_id != device_idx:
                    logger.warning(
                        f"Current GPU ID ({gpu_id}) does not match specified device index ({device_idx})."
                    )
                torch.cuda.reset_max_memory_allocated(gpu_id)
                total_memory = torch.cuda.get_device_properties(gpu_id).total_memory / (
                    1024**3
                )
                allocated_memory = torch.cuda.memory_allocated(gpu_id) / (1024**3)
                logger.log_with_borders(
                    level=logging.INFO,
                    message=f"Using GPU device.\n"
                    f"  - Specified device: '{device}'\n"
                    f"  - Active GPU ID: {gpu_id}\n"
                    f"  - Total Memory: {total_memory:.2f} GB\n"
                    f"  - Initial Allocated Memory: {allocated_memory:.2f} GB",
                    border="|",
                    length=100,
                )
            logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

        # Load model structure and state_dict
        model = load_trained_model(
            model_type=model_type,
            checkpoint_path=model_input_path,
            device=device,
            logger=logger,
        )
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

        # Move test data to device
        if "cuda" in device and torch.cuda.is_available():
            X_test_fcgr = X_test_fcgr.to(device)
            if model_type == "ania":
                if X_test_word is None:
                    raise ValueError("X_test_word must be provided for 'ania' model.")
                X_test_word = X_test_word.to(device)
            y_test = y_test.to(device)
            logger.log_with_borders(
                level=logging.INFO,
                message=f"Moved testing data to '{device}'",
                border="|",
                length=100,
            )
            logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

        # Predict
        with torch.no_grad():
            if model_type == "ania":
                y_pred, _ = model(X_test_fcgr, X_test_word)
            else:
                y_pred, _ = model(X_test_fcgr)
        y_pred = y_pred.cpu().numpy().flatten()
        y_true = y_test.cpu().numpy().flatten()

        # Save predictions
        df_results = df_metadata.copy()
        df_results["Log MIC Value"] = y_true
        df_results["Predicted Log MIC Value"] = y_pred

        # Record start time and resource usage
        start_time = time.time()

        os.makedirs(os.path.dirname(prediction_output_path), exist_ok=True)
        df_results.to_csv(prediction_output_path, index=False)

        # Compute evaluation metrics
        evaluate_dl_predictions(prediction_output_path, model_type, logger)
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

        # Record end time and final GPU memory usage
        total_time = time.time() - start_time
        if "cuda" in device and torch.cuda.is_available() and gpu_id is not None:
            final_allocated_memory = torch.cuda.memory_allocated(gpu_id) / (1024**3)
            final_reserved_memory = torch.cuda.memory_reserved(gpu_id) / (1024**3)
            final_max_memory = torch.cuda.max_memory_allocated(gpu_id) / (1024**3)
            logger.log_with_borders(
                level=logging.INFO,
                message=f"Testing completed for '{model_type}' model:\n"
                f"  • Time: {total_time:.2f}s\n"
                f"  • Final GPU Memory Usage:\n"
                f"    ▸ Allocated: {final_allocated_memory:.2f} GB\n"
                f"    ▸ Reserved: {final_reserved_memory:.2f} GB\n"
                f"    ▸ Max Allocated: {final_max_memory:.2f} GB",
                border="|",
                length=100,
            )
        else:
            logger.log_with_borders(
                level=logging.INFO,
                message=f"Total training completed for '{model_type}':\n"
                f"  • Time: {total_time:.2f}s\n"
                f"  • Running on CPU",
                border="|",
                length=100,
            )
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

        # Log successful save
        logger.log_with_borders(
            level=logging.INFO,
            message=f"Predictions saved:\n'{prediction_output_path}'",
            border="|",
            length=100,
        )
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

    except ValueError:
        logger.exception(msg=f"ValueError in 'test_model()' for '{model_type}'.")
        raise

    except FileNotFoundError:
        logger.exception(msg=f"FileNotFoundError in 'test_model()' for '{model_type}'.")
        raise

    except Exception:
        logger.exception(msg=f"Unexpected error in 'test_model()' for '{model_type}'.")
        raise


# ============================== Pipeline Entry Point ==============================
def run_test_dl_pipeline(
    base_path: str,
    logger: CustomLogger,
    model_type=None,
    device: str = "cuda:0",
) -> None:
    """
    Run the deep learning testing pipeline for multiple strains.

    Parameters
    ----------
    base_path : str
        Absolute base path of the project (used to construct file paths).
    logger : CustomLogger
        Logger for structured logging throughout the pipeline.
    model_type : str, optional
        Type of deep learning model to test ('fcgr_ania', 'word_embedding_ania', or 'ania'). If None or 'all', tests all models.
    device : str, optional
        Device to run the model on (default: 'cuda:0').

    Returns
    -------
    None
        This function performs file I/O and logging side effects, and does not return any value.
    """
    try:
        # Mapping of full strain names to their corresponding suffixes
        strains = {
            "Escherichia coli": "EC",
            "Pseudomonas aeruginosa": "PA",
            "Staphylococcus aureus": "SA",
        }

        # Define models to test
        all_models = ["fcgr_ania", "word_embedding_ania", "ania"]
        if isinstance(model_type, list):
            models_to_test = model_type if "all" not in model_type else all_models
        else:
            models_to_test = (
                [model_type] if model_type and model_type != "all" else all_models
            )

        # Define the target column and metadata columns
        target_column = "Log MIC Value"
        metadata_columns = ["ID", "Sequence", "Targets"]

        # Define feature sets for each model type
        feature_sets = {
            "fcgr_ania": [
                {
                    "name": "CGR",
                    "start_idx": 315,
                    "end_idx": 570,
                    "height": 16,
                    "width": 16,
                    "suffix": "_cgr",
                }
            ],
            "word_embedding_ania": [
                {
                    "name": "WordEmbedding",
                    "llm_name": "BERT-BFD",
                    "height": 64,
                    "width": None,
                    "suffix": "_embed_bert_bfd",
                }
            ],
            "ania": [
                {
                    "name": "CGR",
                    "start_idx": 315,
                    "end_idx": 570,
                    "height": 16,
                    "width": 16,
                    "suffix": "_cgr",
                },
                {
                    "name": "WordEmbedding",
                    "llm_name": "BERT-BFD",
                    "height": 64,
                    "width": None,
                    "suffix": "_embed_bert_bfd",
                },
            ],
        }

        # Loop over each strain type
        for strain_index, (strain, suffix) in enumerate(strains.items(), start=1):
            for model_idx, model in enumerate(models_to_test, start=1):
                test_input_csv_file = os.path.join(
                    base_path, f"data/processed/{suffix}/test.csv"
                )
                test_input_npz_file = (
                    os.path.join(
                        base_path,
                        f"data/processed/{suffix}/test_embed_{feature_sets[model][1]['llm_name']}.npz",
                    )
                    if model == "ania"
                    else (
                        os.path.join(
                            base_path,
                            f"data/processed/{suffix}/test_embed_{feature_sets[model][0]['llm_name']}.npz",
                        )
                        if model == "word_embedding_ania"
                        else None
                    )
                )

                if model == "ania":
                    # Logging: strain section + feature header for ANIA
                    logger.info(msg=f"/ {strain_index}.{model_idx}")
                    logger.add_divider(
                        level=logging.INFO, length=110, border="+", fill="-"
                    )
                    logger.log_with_borders(
                        level=logging.INFO,
                        message=f"Strain - '{strain}' -> Model - '{model}' -> Features - 'CGR + WordEmbedding'",
                        border="|",
                        length=110,
                    )
                    logger.add_divider(
                        level=logging.INFO, length=110, border="+", fill="-"
                    )
                    logger.add_spacer(level=logging.INFO, lines=1)

                    # Extract FCGR features
                    df_metadata, X_test_fcgr, y_test = (
                        extract_cgr_features_and_target_for_dl(
                            file_path=test_input_csv_file,
                            metadata_columns=metadata_columns,
                            target_column=target_column,
                            feature_start_idx=feature_sets[model][0]["start_idx"],
                            feature_end_idx=feature_sets[model][0]["end_idx"],
                            height=feature_sets[model][0]["height"],
                            width=feature_sets[model][0]["width"],
                            logger=logger,
                        )
                    )
                    # Extract Word Embedding features
                    _, X_test_word, _ = extract_word_embedding_features_for_dl(
                        npz_path=test_input_npz_file,  # type: ignore
                        df_path=test_input_csv_file,
                        metadata_columns=metadata_columns,
                        target_column=target_column,
                        height=feature_sets[model][1]["height"],
                        width=feature_sets[model][1]["width"],
                        logger=logger,
                    )
                    logger.add_spacer(level=logging.INFO, lines=1)

                    model_input_path = os.path.join(
                        base_path,
                        f"experiments/models/{suffix}/deep_learning/{model}_combined_finetuned_full.pt",
                    )
                    prediction_output_path = os.path.join(
                        base_path,
                        f"experiments/predictions/{suffix}/deep_learning/{model}_combined.csv",
                    )

                    # Test the ANIA model
                    test_dl_model(
                        df_metadata=df_metadata,
                        X_test_fcgr=X_test_fcgr,
                        X_test_word=X_test_word,
                        y_test=y_test,
                        model_type=model,
                        model_input_path=model_input_path,
                        prediction_output_path=prediction_output_path,
                        logger=logger,
                        device=device,
                    )
                else:
                    for _, feature in enumerate(feature_sets[model], start=1):
                        # Logging: strain section + feature header
                        logger.info(msg=f"/ {strain_index}.{model_idx}")
                        logger.add_divider(
                            level=logging.INFO, length=110, border="+", fill="-"
                        )
                        logger.log_with_borders(
                            level=logging.INFO,
                            message=f"Strain - '{strain}' -> Model - '{model}' -> Feature - '{feature['name']}'",
                            border="|",
                            length=110,
                        )
                        logger.add_divider(
                            level=logging.INFO, length=110, border="+", fill="-"
                        )
                        logger.add_spacer(level=logging.INFO, lines=1)

                        # Extract metadata, features, and target variable
                        if model == "fcgr_ania":
                            df_metadata, X_test_fcgr, y_test = (
                                extract_cgr_features_and_target_for_dl(
                                    file_path=test_input_csv_file,
                                    metadata_columns=metadata_columns,
                                    target_column=target_column,
                                    feature_start_idx=feature["start_idx"],
                                    feature_end_idx=feature["end_idx"],
                                    height=feature["height"],
                                    width=feature["width"],
                                    logger=logger,
                                )
                            )
                            X_test_word = None
                        elif model == "word_embedding_ania":
                            df_metadata, X_test_fcgr, y_test = (
                                extract_word_embedding_features_for_dl(
                                    npz_path=test_input_npz_file,  # type: ignore
                                    df_path=test_input_csv_file,
                                    metadata_columns=metadata_columns,
                                    target_column=target_column,
                                    height=feature["height"],
                                    width=feature["width"],
                                    logger=logger,
                                )
                            )
                            X_test_word = None
                        logger.add_spacer(level=logging.INFO, lines=1)

                        model_input_path = os.path.join(
                            base_path,
                            f"experiments/models/{suffix}/deep_learning/{model}{feature['suffix']}_finetuned_full.pt",
                        )
                        prediction_output_path = os.path.join(
                            base_path,
                            f"experiments/predictions/{suffix}/deep_learning/{model}{feature['suffix']}.csv",
                        )

                        # Test the single-input model
                        test_dl_model(
                            df_metadata=df_metadata,
                            X_test_fcgr=X_test_fcgr,
                            X_test_word=X_test_word,
                            y_test=y_test,
                            model_type=model,
                            model_input_path=model_input_path,
                            prediction_output_path=prediction_output_path,
                            logger=logger,
                            device=device,
                        )

                    # Insert a blank line in the log for readability
                    logger.add_spacer(level=logging.INFO, lines=1)

    except Exception:
        logger.exception(msg="Unexpected error in 'run_test_dl_pipeline()'.")
        raise
