# pylint: disable=line-too-long, import-error, wrong-import-position, broad-exception-caught
"""
ANIA - Project Main Entry Point

This script serves as the centralized command-line interface for executing the ANIA pipeline,
which enables the preprocessing and analysis of antimicrobial peptide (AMP) data.

Supported pipeline stages include:
----------------------------------
- Data Preprocessing (collecting, cleaning)

Users can run specific stages by specifying pipeline targets.
-----
Users can run specific stages via:
    python main.py --stage collect
    python main.py --stage preprocess
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
}


# ============================== Pipeline Dispatcher ==============================
def dispatch_stage(stage: str) -> None:
    """
    Dispatch execution to the appropriate pipeline stage using lazy import.

    Parameters
    ----------
    stage : str
        The pipeline stage to execute.
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
            "\n"
            "Stage combinations:\n"
            "  --preprocess  Shortcut for: collect → clean\n"
            "  --all         Run all supported pipeline stages sequentially\n"
            "\n"
            "Examples:\n"
            "  python main.py --stage collect\n"
            "  python main.py --stage clean\n"
            "  python main.py --preprocess\n"
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
        "--preprocess",
        action="store_true",
        help="",
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
        elif args.stage:
            stages_to_run = args.stage
        else:
            parser.error("You must specify either --stage or --all.")

        # Dispatch each selected stage
        for stage in stages_to_run:
            dispatch_stage(stage=stage.lower())

        # Final success message
        print("Pipeline execution completed successfully.")
        sys.exit(0)

    except Exception as e:
        print(f"[Pipeline Error] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
