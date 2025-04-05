# pylint: disable=line-too-long, import-error, wrong-import-position, broad-exception-caught
"""
ANIA - Project Main Entry Point

This script serves as the centralized command-line interface for executing the ANIA pipeline,
which enables the preprocessing and analysis of antimicrobial peptide (AMP) data.
"""
# ============================== Standard Library Imports ==============================
import argparse
import importlib
import logging
import os
import sys

# ============================== Project Root Path Setup ==============================
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
LOGGING_PATH = os.path.join(BASE_PATH, "src/utils/logging_toolkit/src/python")

# Fix Python Path
if BASE_PATH not in sys.path:
    sys.path.append(BASE_PATH)
if LOGGING_PATH not in sys.path:
    sys.path.append(LOGGING_PATH)

# ============================== Project-Specific Imports ==============================
# Logging configuration and custom logger
from setup_logging import setup_logging

# ============================== Stage Configuration ==============================
SUPPORTED_STAGES = {
    "collect": {
        "title": "Data Collecting",
        "import_path": "src.data.collect.run_collect_pipeline",
    },
    "clean": {
        "title": "Data Cleaning",
        "import_path": "src.data.clean.run_clean_pipeline",
    },
    "ifeature": {
        "title": "Feature Extraction (iFeature)",
        "import_path": "src.features.ifeature_encoding.run_ifeature_pipeline",
    },
    "cgr": {
        "title": "Feature Extraction (Chaos game representation)",
        "import_path": "src.features.cgr_encoding.run_cgr_pipeline",
    },
    "train_ml": {
        "title": "Train Machine Learning Model",
        "import_path": "src.models.machine_learning.train.run_train_ml_pipeline",
    },
    "test_ml": {
        "title": "Test Machine Learning Model",
        "import_path": "src.models.machine_learning.test.run_test_ml_pipeline",
    },
}


# ============================== Pipeline Dispatcher ==============================
def dispatch_stage(stage: str, args) -> None:
    """
    Dispatch execution to the appropriate pipeline stage using lazy import.

    Parameters
    ----------
    stage : str
        The pipeline stage to execute.
    args : argparse.Namespace
        Command-line arguments containing stage-specific options.
    """
    # Setup dedicated log file for this stage
    log_config_file = os.path.join(BASE_PATH, "configs/general_logging.json")
    stage_log_path = os.path.join(BASE_PATH, f"logs/{stage}_stage.log")
    output_dir = os.path.dirname(stage_log_path)
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logging(
        input_config_file=log_config_file,
        output_log_path=stage_log_path,
        logger_name=f"{stage}_logger",
        handler_name="general",
    )

    # Validate stage
    if stage not in SUPPORTED_STAGES:
        available = ", ".join(SUPPORTED_STAGES.keys())
        logger.error(f"Invalid stage '{stage}'. Available stages: {available}.")
        raise ValueError(f"Unknown stage '{stage}'.")

    # Lazy load stage function
    stage_info = SUPPORTED_STAGES[stage]
    module_path, func_name = stage_info["import_path"].rsplit(".", 1)
    module = importlib.import_module(module_path)
    stage_func = getattr(module, func_name)

    # Log and run
    logger.log_title(
        f"Running Stage: '{stage_info['title']}'",
        level=logging.INFO,
        length=40,
        border="=",
    )
    if stage == "train_ml":
        if not args.model_type:
            logger.error("Missing '--model_type' for 'train_ml' stage.")
            raise ValueError(
                "Please specify '--model_type' (e.g., 'linear', 'random_forest', 'svm', 'xgboost', or 'all') for 'train_ml' stage."
            )
        stage_func(
            base_path=BASE_PATH,
            model_type=args.model_type,
            n_jobs=args.n_jobs,
            random_state=args.random_state,
            cv=args.cv,
            loss_function=args.loss_function,
            logger=logger,
        )
    elif stage == "test_ml":
        if not args.model_type:
            logger.error("Missing '--model_type' for 'test_ml' stage.")
            raise ValueError(
                "Please specify '--model_type' (e.g., 'linear', 'random_forest', 'svm', 'xgboost', or 'all') for 'test_ml' stage."
            )
        stage_func(
            base_path=BASE_PATH,
            logger=logger,
            model_type=args.model_type,
        )
    else:
        stage_func(base_path=BASE_PATH, logger=logger)


# ============================== Main Entry ==============================
def main():
    """
    Main CLI entry point for the AMP-MIC pipeline.
    Parses CLI arguments and routes execution to the selected pipeline stages.
    """
    # -------------------- Argument Parser --------------------
    parser = argparse.ArgumentParser(
        description=(
            "ANIA - An Inception-Attention Network for Predicting the Minimum Inhibitory Concentration (MIC) of Antimicrobial Peptides\n"
            "\n"
            "Pipeline stages:\n"
            "  collect       Run data collection from external AMP databases\n"
            "  clean         Clean and aggregate data for downstream modeling\n"
            "  ifeature      Extract iFeature-based descriptors (AAC, PAAC, CTDD, GAAC)\n"
            "  cgr           Generate CGR features at multiple resolutions (8x8, 16x16, ...)\n"
            "  train_ml      Train machine learning models on iFeature and CGR features (specify --model_type)\n"
            "  test_ml       Test trained machine learning models on test data (specify --model_type)\n"
            "\n"
            "Stage combinations:\n"
            "  --preprocess  Shortcut for: collect → clean\n"
            "  --encoding    Shortcut for: ifeature → cgr\n"
            "  --machine_learning  Shortcut for: train_ml → test_ml\n"
            "  --all         Run all supported pipeline stages sequentially\n"
            "\n"
            "Examples:\n"
            "  python main.py --stage collect\n"
            "  python main.py --stage train_ml --model_type linear random_forest --n_jobs -1 --random_state 42 --cv 5\n"
            "  python main.py --stage test_ml --model_type linear random_forest\n"
            "  python main.py --preprocess\n"
            "  python main.py --encoding\n"
            "  python main.py --machine_learning --model_type linear random_forest\n"
            "  python main.py --all"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--stage",
        type=str,
        nargs="+",
        choices=list(SUPPORTED_STAGES.keys()),
        help="Pipeline stage(s) to run. Choose one or more of: "
        + ", ".join(SUPPORTED_STAGES.keys()),
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all available pipeline stages sequentially.",
    )

    parser.add_argument(
        "--preprocess", action="store_true", help="Run collect → clean."
    )

    parser.add_argument("--encoding", action="store_true", help="Run ifeature → cgr.")

    parser.add_argument(
        "--machine_learning", action="store_true", help="Run train_ml → test_ml."
    )

    parser.add_argument(
        "--model_type",
        type=str,
        nargs="+",
        choices=[
            "linear",
            "lasso",
            "ridge",
            "elastic_net",
            "random_forest",
            "svm",
            "xgboost",
            "gradient_boosting",
            "all",
        ],
        help="Model type(s) for train_ml stage (e.g., 'linear random_forest' or 'all' for all models).",
    )

    parser.add_argument(
        "--n_jobs",
        type=int,
        default=-1,
        help="Number of CPU cores for training (-1 for all available cores).",
    )

    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for reproducibility in training (default is 42).",
    )

    parser.add_argument(
        "--cv",
        type=int,
        default=5,
        help="Number of cross-validation folds for GridSearchCV in training (default is 5).",
    )

    parser.add_argument(
        "--loss_function",
        type=str,
        default="neg_mean_squared_error",
        choices=["neg_mean_squared_error", "neg_mean_absolute_error"],
        help="Loss function for GridSearchCV scoring ('neg_mean_squared_error', 'neg_mean_absolute_error'). Default is 'neg_mean_squared_error'.",
    )

    args = parser.parse_args()

    # -------------------- Stage Execution --------------------
    try:

        # Determine which stages to execute
        stages_to_run = []
        if args.all:
            stages_to_run = list(SUPPORTED_STAGES.keys())
        elif args.preprocess:
            stages_to_run = ["collect", "clean"]
        elif args.encoding:
            stages_to_run = ["ifeature", "cgr"]
        elif args.machine_learning:
            stages_to_run = ["train_ml", "test_ml"]
        elif args.stage:
            stages_to_run = args.stage
        else:
            parser.error("You must specify either --stage or --all.")

        # Dispatch each selected stage
        for stage in stages_to_run:
            dispatch_stage(stage.lower(), args)

        # Final success message
        print("Pipeline execution completed successfully.")
        sys.exit(0)

    except Exception as e:
        print(f"[Pipeline Error] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
