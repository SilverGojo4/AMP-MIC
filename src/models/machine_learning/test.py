# pylint: disable=line-too-long, import-error, wrong-import-position, too-many-locals, too-many-statements, invalid-name, too-many-arguments, too-many-positional-arguments
"""
Machine Learning Model Testing Module

This module provides functionality for testing baseline machine learning models in the AMP-MIC project
to predict the Minimum Inhibitory Concentration (MIC) of antimicrobial peptides (AMPs). It supports
evaluation of Linear Regression, Lasso, Ridge, Elastic Net, Random Forest, SVM, XGBoost, and Gradient Boosting
models on test data, loading trained models and computing performance metrics.
"""
# ============================== Standard Library Imports ==============================
import logging
import os
import sys
import time
import tracemalloc

# ============================== Third-Party Library Imports ==============================
import joblib
import numpy as np
import pandas as pd
import psutil
from sklearn.exceptions import NotFittedError
from sklearn.metrics import mean_squared_error, r2_score

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

# Machine Learning Utility Functions
from src.models.machine_learning.utils import extract_features_and_target


# ============================== Custom Function ==============================
def evaluate_predictions(
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
        Type of model to test ('linear', 'random_forest', 'svm', 'xgboost').
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
        logger.exception(msg="FileNotFoundError in 'evaluate_predictions()'.")
        raise

    except KeyError:
        logger.exception(msg="KeyError in 'evaluate_predictions()'.")
        raise

    except Exception:
        logger.exception(msg="Unexpected error in 'evaluate_predictions()'.")
        raise


def test_model(
    df_metadata: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_type: str,
    model_input_path: str,
    prediction_output_path: str,
    logger: CustomLogger,
) -> None:
    """
    Test a trained machine learning model on test data and save predictions.

    Parameters
    ----------
    df_metadata : pd.DataFrame
        DataFrame containing metadata (e.g., ID, Sequence, Targets).
    X_test : pd.DataFrame
        Test feature set.
    y_test : pd.Series
        Test target variable.
    model_type : str
        Type of model to test ('linear', 'random_forest', 'svm', 'xgboost').
    model_input_path : str
        Path to the trained model file.
    prediction_output_path : str
        Path to save the prediction results.
    logger : CustomLogger
        Logger instance for tracking progress and errors.

    Returns
    -------
    Optional[dict]
        Dictionary containing performance metrics, or None if testing fails.
    """
    try:
        # Log training start
        logger.info(msg=f"/ Task: Test '{model_type}' model")
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

        # Record start time and resource usage
        tracemalloc.start()
        start_time = time.time()
        cpu_usage_start = psutil.cpu_percent(interval=1)
        ram_usage_start = psutil.virtual_memory().percent

        # Load trained model
        model = joblib.load(model_input_path)
        logger.log_with_borders(
            level=logging.INFO,
            message=f"Successfully loaded trained '{model_type}' model from\n'{model_input_path}'",
            border="|",
            length=100,
        )
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

        # Make predictions
        y_pred = model.predict(X_test)

        # Save predictions first (before computing metrics)
        df_results = df_metadata.copy()
        df_results["Log MIC Value"] = y_test
        df_results["Predicted Log MIC Value"] = y_pred

        os.makedirs(os.path.dirname(prediction_output_path), exist_ok=True)
        df_results.to_csv(prediction_output_path, index=False)

        # Compute evaluation metrics
        evaluate_predictions(prediction_output_path, model_type, logger)
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

        # Record end time and resource usage
        total_time = time.time() - start_time
        cpu_usage_end = psutil.cpu_percent(interval=1)
        ram_usage_end = psutil.virtual_memory().percent
        _, peak_memory = tracemalloc.get_traced_memory()  # Get peak memory in bytes
        peak_memory_mb = peak_memory / 1024 / 1024  # Convert to MB
        tracemalloc.stop()
        logger.log_with_borders(
            level=logging.INFO,
            message=f"Testing completed for '{model_type}' model:\n"
            f"  • Time: {total_time:.2f}s\n"
            f"  • Peak RAM Usage: {peak_memory_mb:.2f} MB\n"
            f"  • CPU: {cpu_usage_start:.2f}% → {cpu_usage_end:.2f}%\n"
            f"  • RAM: {ram_usage_start:.2f}% → {ram_usage_end:.2f}%",
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

    except NotFittedError:
        logger.exception(msg=f"NotFittedError in 'test_model()' for '{model_type}'.")
        raise

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
def run_test_ml_pipeline(
    base_path: str,
    logger: CustomLogger,
    model_type=None,
) -> None:
    """
    Run the machine learning training pipeline for multiple strains.

    Parameters
    ----------
    base_path : str
        Absolute base path of the project (used to construct file paths).
    logger : CustomLogger
        Logger for structured logging throughout the pipeline.
    model_type : str or list, optional
        Specific model type(s) to train ('linear', 'random_forest', 'svm', 'xgboost'). If None or 'all', trains all models.

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

        # Define models to train
        all_models = [
            "linear",
            "lasso",
            "ridge",
            "elastic_net",
            "random_forest",
            "svm",
            "xgboost",
            "gradient_boosting",
        ]
        if isinstance(model_type, list):
            models_to_train = model_type if "all" not in model_type else all_models
        else:
            models_to_train = (
                [model_type] if model_type and model_type != "all" else all_models
            )

        # Define the target column and metadata columns
        target_column = "Log MIC Value"
        metadata_columns = ["ID", "Sequence", "Targets"]

        # Define a list of feature sets to process for each strain in the training pipeline.
        feature_sets = [
            {
                "name": "iFeature",
                "start_idx": 6,
                "end_idx": 250,
                "standardize": True,
                "suffix": "",
            },
            {
                "name": "CGR",
                "start_idx": 315,
                "end_idx": 570,
                "standardize": False,
                "suffix": "_cgr",
            },
        ]

        # Loop over each strain type
        for strain_index, (strain, suffix) in enumerate(strains.items(), start=1):

            # Define dataset paths
            test_input_csv_file = os.path.join(
                base_path, f"data/processed/{suffix}/test.csv"
            )

            for feature_idx, feature in enumerate(feature_sets, start=1):

                # -------------------- Logging: strain section + feature header --------------------
                logger.info(msg=f"/ {strain_index}.{feature_idx}")
                logger.add_divider(level=logging.INFO, length=70, border="+", fill="-")
                logger.log_with_borders(
                    level=logging.INFO,
                    message=f"Strain - '{strain}' -> Feature - '{feature['name']}'",
                    border="|",
                    length=70,
                )
                logger.add_divider(level=logging.INFO, length=70, border="+", fill="-")
                logger.add_spacer(level=logging.INFO, lines=1)

                # Extract metadata, features, and target variable from the testing CSV file for the current strain and feature set.
                df_metadata, X_test, y_test = extract_features_and_target(
                    file_path=test_input_csv_file,
                    metadata_columns=metadata_columns,
                    target_column=target_column,
                    feature_start_idx=feature["start_idx"],
                    feature_end_idx=feature["end_idx"],
                    logger=logger,
                    standardize=feature["standardize"],
                )
                logger.add_spacer(level=logging.INFO, lines=1)

                # Loop over each model type
                for _, model in enumerate(models_to_train, start=1):
                    model_input_path = os.path.join(
                        base_path,
                        f"experiments/models/{suffix}/machine_learning/{model}{feature['suffix']}.pkl",
                    )
                    prediction_output_path = os.path.join(
                        base_path,
                        f"experiments/predictions/{suffix}/machine_learning/{model}{feature['suffix']}.csv",
                    )
                    test_model(
                        df_metadata=df_metadata,
                        X_test=X_test,
                        y_test=y_test,
                        model_type=model,
                        model_input_path=model_input_path,
                        prediction_output_path=prediction_output_path,
                        logger=logger,
                    )

                    # Insert a blank line in the log for readability
                    logger.add_spacer(level=logging.INFO, lines=1)

    except Exception:
        logger.exception(msg="Unexpected error in 'run_test_ml_pipeline()'.")
        raise
