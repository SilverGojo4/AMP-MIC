# pylint: disable=line-too-long, import-error, wrong-import-position, too-many-locals, too-many-statements, invalid-name, too-many-arguments, too-many-positional-arguments
"""
Machine Learning Model Training Module

This module provides functionality for training baseline machine learning models in the AMP-MIC project
to predict the Minimum Inhibitory Concentration (MIC) of antimicrobial peptides (AMPs). It supports
training of Linear Regression, Lasso, Ridge, Elastic Net, Random Forest, SVM, XGBoost, and Gradient Boosting
models using GridSearchCV for hyperparameter tuning (except Linear Regression), and saves the trained models for
later evaluation.
"""
# ============================== Standard Library Imports ==============================
import logging
import os
import sys
import time
import tracemalloc
from typing import Optional

# ============================== Third-Party Library Imports ==============================
import joblib
import pandas as pd
import psutil
from sklearn.model_selection import GridSearchCV, ParameterGrid

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
from src.models.machine_learning.architecture import get_model
from src.models.machine_learning.utils import (
    extract_features_and_target,
    get_hyperparameter_settings,
    read_json_config,
)


# ============================== Custom Function ==============================
def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str,
    hyperparams_config_path: str,
    model_output_path: str,
    logger: CustomLogger,
    n_jobs: int = -1,
    random_state: int = 42,
    cv: int = 5,
    loss_function: str = "neg_mean_squared_error",
) -> Optional[object]:
    """
    Train a machine learning model with optional GridSearchCV and save it.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature set.
    y_train : pd.Series
        Training target variable.
    model_type : str
        Type of model to train ('linear', 'random_forest', 'svm', 'xgboost').
    hyperparams_config_path : str
        Path to the JSON file containing hyperparameter ranges.
    model_output_path : str
        Path to save the trained model.
    logger : CustomLogger
        Logger instance for tracking progress and errors.
    n_jobs : int
        Number of CPU cores to use (-1 for all available cores).
    random_state : int, optional
        Random seed for reproducibility. Default is 42.
    cv : int, optional
        Number of cross-validation folds for GridSearchCV. Default is 5.
    loss_function : str, optional
        Loss function for GridSearchCV scoring ('neg_mean_squared_error', 'neg_mean_absolute_error'). Default is 'neg_mean_squared_error'.

    Returns
    -------
    Optional[object]
        Trained model instance, or None if training fails.
    """
    try:
        # Log training start
        logger.info(
            msg=f"/ Task: Train '{model_type}' model with loss function '{loss_function}'"
        )
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

        # Record start time and resource usage
        tracemalloc.start()
        start_time = time.time()
        cpu_usage_start = psutil.cpu_percent(interval=1)
        ram_usage_start = psutil.virtual_memory().percent

        # Initialize model
        model = get_model(model_type, logger, n_jobs, random_state)
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

        # Train model (with GridSearchCV for non-linear models)
        if model_type == "linear":
            logger.log_with_borders(
                level=logging.INFO,
                message="Training 'linear regression' without 'hyperparameter tuning'.",
                border="|",
                length=100,
            )
            logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")
            model.fit(X_train, y_train)
        else:
            # Load hyperparameters
            config = read_json_config(hyperparams_config_path, logger)
            param_grid = get_hyperparameter_settings(config, model_type, logger)
            loss_name = "MSE" if loss_function == "neg_mean_squared_error" else "MAE"
            logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

            # Calculate total number of training iterations
            num_parameters = len(param_grid)
            total_combinations = len(list(ParameterGrid(param_grid)))
            total_iterations = total_combinations * cv
            logger.log_with_borders(
                level=logging.INFO,
                message=f"Performing 'GridSearchCV' for '{model_type}':\n"
                f"  • Total number of parameters: {num_parameters}\n"
                f"  • Number of parameter combinations: {total_combinations}\n"
                f"  • Cross-validation folds: {cv}\n"
                f"  • Total training iterations: {total_iterations}",
                border="|",
                length=100,
            )
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=cv,
                scoring=loss_function,
                n_jobs=n_jobs,
                verbose=2,
            )
            grid_search.fit(X_train, y_train)

            # Log CV results
            logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")
            cv_results = grid_search.cv_results_
            best_idx = grid_search.best_index_
            best_params = cv_results["params"][best_idx]
            mean_score = -cv_results["mean_test_score"][best_idx]
            std_score = cv_results["std_test_score"][best_idx]
            model = grid_search.best_estimator_

            # Log best hyperparameters
            logger.log_with_borders(
                level=logging.INFO,
                message=f"Best hyperparameters for '{model_type}':\n"
                f"  • Parameters:\n     {best_params}\n"
                f"  • Mean {loss_name}: {mean_score:.4f}\n"
                f"  • Std {loss_name}: {std_score:.4f}",
                border="|",
                length=100,
            )
            logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

        # Record end time and resource usage
        total_time = time.time() - start_time
        cpu_usage_end = psutil.cpu_percent(interval=1)
        ram_usage_end = psutil.virtual_memory().percent
        cpu_cores_used = "'All available'" if n_jobs == -1 else str(n_jobs)
        _, peak_memory = tracemalloc.get_traced_memory()  # Get peak memory in bytes
        peak_memory_mb = peak_memory / 1024 / 1024  # Convert to MB
        tracemalloc.stop()
        logger.log_with_borders(
            level=logging.INFO,
            message=f"Training completed for '{model_type}' model:\n"
            f"  • Time: {total_time:.2f}s\n"
            f"  • CPU Cores Used: {cpu_cores_used}\n"
            f"  • Peak RAM Usage: {peak_memory_mb:.2f} MB\n"
            f"  • CPU: {cpu_usage_start:.2f}% → {cpu_usage_end:.2f}%\n"
            f"  • RAM: {ram_usage_start:.2f}% → {ram_usage_end:.2f}%",
            border="|",
            length=100,
        )
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

        # Save trained model
        os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
        joblib.dump(model, model_output_path)
        logger.log_with_borders(
            level=logging.INFO,
            message=f"Successfully saved trained model to\n'{model_output_path}'",
            border="|",
            length=100,
        )
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

        return model

    except ValueError:
        logger.exception(msg=f"ValueError in 'train_model()' for '{model_type}'.")
        raise

    except FileNotFoundError:
        logger.exception(
            msg=f"FileNotFoundError in 'train_model()' for '{model_type}'."
        )
        raise

    except Exception:
        logger.exception(msg=f"Unexpected error in 'train_model()' for '{model_type}'.")
        raise


# ============================== Pipeline Entry Point ==============================
def run_train_ml_pipeline(
    base_path: str,
    logger: CustomLogger,
    model_type=None,
    n_jobs: int = -1,
    random_state: int = 42,
    cv: int = 5,
    loss_function: str = "neg_mean_squared_error",
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
    n_jobs : int
        Number of CPU cores to use (-1 for all available cores).
    random_state : int, optional
        Random seed for reproducibility. Default is 42.
    cv : int, optional
        Number of cross-validation folds for GridSearchCV. Default is 5.
    loss_function : str, optional
        Loss function for GridSearchCV scoring ('neg_mean_squared_error', 'neg_mean_absolute_error'). Default is 'neg_mean_squared_error'.

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

        # Path to hyperparameter configuration
        hyperparams_config_path = os.path.join(
            base_path, "configs/ML/hyperparameters.json"
        )

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
            train_input_csv_file = os.path.join(
                base_path, f"data/processed/{suffix}/train.csv"
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

                # Extract metadata, features, and target variable from the training CSV file for the current strain and feature set.
                _, X_train, y_train = extract_features_and_target(
                    file_path=train_input_csv_file,
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
                    model_output_path = os.path.join(
                        base_path,
                        f"experiments/models/{suffix}/ML/{model}{feature['suffix']}.pkl",
                    )
                    train_model(
                        X_train=X_train,
                        y_train=y_train,
                        model_type=model,
                        hyperparams_config_path=hyperparams_config_path,
                        model_output_path=model_output_path,
                        logger=logger,
                        n_jobs=n_jobs,
                        random_state=random_state,
                        cv=cv,
                        loss_function=loss_function,
                    )

                    # Insert a blank line in the log for readability
                    logger.add_spacer(level=logging.INFO, lines=1)

    except Exception:
        logger.exception(msg="Unexpected error in 'run_train_ml_pipeline()'.")
        raise
