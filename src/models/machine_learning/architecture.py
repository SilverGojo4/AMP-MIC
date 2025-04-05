# pylint: disable=line-too-long, import-error, wrong-import-position, too-many-locals, too-many-statements
"""
Machine Learning Model Architecture Module

This module provides functionality for initializing baseline machine learning models used in the AMP-MIC project
to predict the Minimum Inhibitory Concentration (MIC) of antimicrobial peptides (AMPs). It supports multiple
standard ML algorithms, including Linear Regression, Random Forest, Support Vector Machine (SVM), and XGBoost,
which serve as foundational models for regression tasks on AMP data.

The initialization pipeline:
1. Accepts a model type identifier and optional parameters (e.g., number of CPU cores, random seed).
2. Configures and returns an instance of the specified model with appropriate settings.
3. Logs initialization steps for traceability and debugging.

The core function in this module is:
- `get_model()`: Initializes and returns a machine learning model instance based on the specified type.
"""
# ============================== Standard Library Imports ==============================
import logging
import os
import sys
from typing import Union

# ============================== Third-Party Library Imports ==============================
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.svm import SVR
from xgboost import XGBRegressor

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
def get_model(
    model_type: str,
    logger: CustomLogger,
    n_jobs: int = -1,
    random_state: int = 42,
) -> Union[LinearRegression, RandomForestRegressor, SVR, XGBRegressor]:
    """
    Return a machine learning model instance based on type.

    Parameters
    ----------
    model_type : str
        Type of model to initialize ('linear', 'random_forest', 'svm', 'xgboost').
    logger : CustomLogger
        Logger instance for tracking progress and errors.
    n_jobs : int, optional
        Number of CPU cores to use (-1 for all available cores). Default is -1.
    random_state : int, optional
        Random seed for reproducibility. Default is 42.

    Returns
    -------
    Union[LinearRegression, RandomForestRegressor, SVR, XGBRegressor]
        Initialized model instance.
    """
    try:
        logger.log_with_borders(
            level=logging.INFO,
            message=f"Initializing '{model_type}' model.",
            border="|",
            length=100,
        )

        if model_type == "linear":
            model = LinearRegression(n_jobs=n_jobs)
        elif model_type == "lasso":
            model = Lasso(random_state=random_state)
        elif model_type == "ridge":
            model = Ridge(random_state=random_state)
        elif model_type == "elastic_net":
            model = ElasticNet(random_state=random_state)
        elif model_type == "random_forest":
            model = RandomForestRegressor(random_state=random_state, n_jobs=n_jobs)
        elif model_type == "svm":
            model = SVR()
        elif model_type == "xgboost":
            model = XGBRegressor(
                objective="reg:squarederror", n_jobs=n_jobs, random_state=random_state
            )
        elif model_type == "gradient_boosting":
            model = GradientBoostingRegressor(random_state=random_state)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        logger.log_with_borders(
            level=logging.INFO,
            message=f"Successfully initialized '{model_type}' model.",
            border="|",
            length=100,
        )

        return model

    except Exception:
        logger.exception(msg=f"Unexpected error in 'get_model()' for '{model_type}'.")
        raise
