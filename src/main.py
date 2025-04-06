# pylint: disable=line-too-long, import-error, wrong-import-position, broad-exception-caught
"""
ANIA - Project Main Entry Point

This script serves as the centralized command-line interface (CLI) for executing the ANIA pipeline,
including preprocessing, feature encoding, classical machine learning, and deep learning stages.

Supported stages include:
  - Data collection & cleaning
  - iFeature and CGR feature extraction
  - Word embedding-based sequence encoding
  - Training and testing of ML models (e.g., SVM, XGBoost)
  - Training and fine-tuning of deep learning models (e.g., FCGRANIA)

Each stage is modular and can be executed independently or in combination via command-line flags.

Usage Example:
  python main.py --stage word_embedding
  python main.py --encoding
  python main.py --machine_learning --model_type svm xgboost
  python main.py --all
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
    "word_embedding": {
        "title": "Feature Extraction (Word Embedding)",
        "import_path": "src.features.word_embedding.run_word_embedding_pipeline",
    },
    "train_dl": {
        "title": "Train Deep Learning Model",
        "import_path": "src.models.deep_learning.train.run_train_dl_pipeline",
    },
    "test_dl": {
        "title": "Test Deep Learning Model",
        "import_path": "src.models.deep_learning.test.run_test_dl_pipeline",
    },
    "fine_tune": {
        "title": "Fine-tune Deep Learning Model",
        "import_path": "src.models.deep_learning.fine_tune.run_fine_tune_pipeline",
    },
}


# ============================== Pipeline Dispatcher ==============================
def dispatch_stage(stage: str, args) -> None:
    """
    Dispatch execution to the appropriate pipeline stage using lazy import.

    This function dynamically imports and executes the pipeline stage function
    based on user input. It supports both classical ML (train_ml, test_ml)
    and deep learning (train_dl) stages, handling stage-specific arguments accordingly.

    Parameters
    ----------
    stage : str
        The pipeline stage to execute.
    args : argparse.Namespace
        Parsed command-line arguments containing stage-specific options.
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
    elif stage == "train_dl":
        if not args.model_type:
            logger.error("Missing '--model_type' for 'train_dl' stage.")
            raise ValueError(
                "Please specify '--model_type' (e.g., 'fcgr_ania') for 'train_dl' stage."
            )
        stage_func(
            base_path=BASE_PATH,
            model_type=args.model_type,
            train_split=args.train_split,
            patience=args.patience,
            random_search=args.random_search,
            num_random_samples=args.num_random_samples,
            device=args.device,
            logger=logger,
        )
    elif stage == "test_dl":
        if not args.model_type:
            logger.error("Missing '--model_type' for 'test_dl' stage.")
            raise ValueError(
                "Please specify '--model_type' (e.g., 'fcgr_ania') for 'test_dl' stage."
            )
        stage_func(
            base_path=BASE_PATH,
            model_type=args.model_type,
            device=args.device,
            logger=logger,
        )
    elif stage == "fine_tune":
        if not args.model_type:
            logger.error("Missing '--model_type' for 'fine_tune' stage.")
            raise ValueError(
                "Please specify '--model_type' (e.g., 'fcgr_ania') for 'fine_tune' stage."
            )
        stage_func(
            base_path=BASE_PATH,
            model_type=(
                args.model_type[0]
                if isinstance(args.model_type, list)
                else args.model_type
            ),
            device=args.device,
            ft_epochs=args.ft_epochs,
            freeze_inception=args.freeze_inception,
            logger=logger,
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
            "  collect         Run data collection from external AMP databases\n"
            "  clean           Clean and aggregate data for downstream modeling\n"
            "  ifeature        Extract iFeature-based descriptors (AAC, PAAC, CTDD, GAAC)\n"
            "  cgr             Generate CGR features at multiple resolutions (8x8, 16x16, ...)\n"
            "  train_ml        Train classical ML models on iFeature/CGR features (requires --model_type)\n"
            "  test_ml         Test classical ML models on evaluation data (requires --model_type)\n"
            "  word_embedding  Encode sequences using precomputed amino acid letter embeddings\n"
            "  train_dl        Train deep learning models using CGR features (e.g., FCGRANIA)\n"
            "\n"
            "Stage combinations:\n"
            "  --preprocess        Shortcut for: collect → clean\n"
            "  --encoding          Shortcut for: ifeature → cgr\n"
            "  --machine_learning  Shortcut for: train_ml → test_ml\n"
            "  --all               Run all supported pipeline stages sequentially\n"
            "\n"
            "Examples:\n"
            "  python main.py --stage collect\n"
            "  python main.py --stage train_ml --model_type linear random_forest --n_jobs -1 --random_state 42 --cv 5\n"
            "  python main.py --stage test_ml --model_type linear random_forest\n"
            "  python main.py --stage train_dl --model_type fcgr_ania --random_search --num_random_samples 30\n"
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
        help=(
            "Model type(s) to run. Depending on the stage, supported values include:\n"
            "  • ML  → linear, lasso, ridge, elastic_net, random_forest, svm, xgboost, gradient_boosting, all\n"
            "  • DL  → fcgr_ania\n"
        ),
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

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use for deep learning training (e.g., 'cuda:0', 'cuda:1', or 'cpu'). Default is 'cuda:0'.",
    )

    parser.add_argument(
        "--random_search",
        action="store_true",
        help="(DL only) Use Random Search instead of full Grid Search for deep learning model training.",
    )

    parser.add_argument(
        "--num_random_samples",
        type=int,
        default=50,
        help="(DL only) Number of random hyperparameter combinations to sample when using Random Search (default: 50).",
    )

    parser.add_argument(
        "--train_split",
        type=float,
        default=0.8,
        help="(DL only) Train/validation split ratio for deep learning model (default: 0.8).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="(DL only) Early stopping patience for deep learning model (default: 10).",
    )

    parser.add_argument(
        "--ft_epochs",
        type=int,
        default=10,
        help="(DL only) Number of fine-tuning epochs (default: 10).",
    )

    parser.add_argument(
        "--freeze_inception",
        action="store_true",
        help="(DL only) Freeze the Inception module during fine-tuning (default: False).",
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
