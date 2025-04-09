# pylint: disable=line-too-long, import-error, wrong-import-position, too-many-locals, too-many-statements
"""
This module handles the cleaning, filtering, and formatting of antimicrobial peptide (AMP) data
after integration. It supports statistical filtering (e.g., Z-score), aggregation of duplicate
sequences, and export of curated datasets in both CSV and FASTA formats. The module also enables
stratified splitting of data for downstream model training.

The processing pipeline:
1. Filters out invalid or extreme 'Log MIC Value' entries using Z-score.
2. Aggregates repeated sequences by selecting the minimum 'Log MIC Value'.
3. Removes duplicate entries and unnecessary columns.
4. Saves the cleaned dataset and writes sequences in FASTA format.
5. Splits the final dataset into training and testing sets (with class balance).

The core functions in this module are:
- `count_sequence_occurrences()`: Counts how many times each sequence appears.
- `count_sequences()`: Categorizes sequences into unique, duplicate, and all, with logging.
- `write_fasta_file()`: Writes sequence data to FASTA format.
- `split_and_save_dataset()`: Splits the dataset and exports CSV and FASTA files.
- `run_clean_pipeline()`: Entry point for cleaning the integrated datasets for each strain.

The `run_clean_pipeline()` function serves as the entry point for running the entire cleaning
pipeline. It loads each strain-specific dataset, applies filtering and aggregation, and saves
the cleaned output for further analysis or modeling.
"""
# ============================== Standard Library Imports ==============================
import logging
import os
import sys
from typing import Dict, List

# ============================== Third-Party Library Imports ==============================
import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.model_selection import train_test_split

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
def count_sequence_occurrences(
    df: pd.DataFrame,
    logger: CustomLogger,
) -> Dict[str, int]:
    """
    Count the occurrences of each sequence and log the number of unique sequences.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame containing the input sequences.
    logger : CustomLogger
        Logger instance for tracking progress and errors.

    Returns
    -------
    Dict[str, int]
        A dictionary with sequences as keys and their occurrence counts as values.
    """
    try:
        # Count the occurrences of each sequence
        sequence_count = df["Sequence"].value_counts().to_dict()
        logger.log_with_borders(
            level=logging.INFO,
            message=f"Counted {len(sequence_count)} unique sequences in 'Sequence'.",
            border="|",
            length=100,
        )

        return sequence_count

    except Exception:
        logger.exception(msg="Unexpected error in 'count_sequence_occurrences()'.")
        raise


def count_sequences(
    df: pd.DataFrame,
    dataset_name: str,
    logger: CustomLogger,
) -> Dict[str, List[str]]:
    """
    Process sequences by categorizing them into unique, duplicate, and all sequences.
    Logs the number of each category.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame containing the input sequences.
    dataset_name : str
        The name of the dataset being processed.
    logger : CustomLogger
        Logger instance for tracking progress and errors.

    Returns
    -------
    Dict[str, List[str]]
        - 'unique': Sequences appearing exactly once in the dataset.
        - 'duplicate': Sequences appearing two or more times.
        - 'once': All unique sequences, regardless of repetition count.
    """
    steps = {
        "extracting 'unique' sequences": lambda sequence_count: [
            sequence for sequence, count in sequence_count.items() if count == 1
        ],
        "extracting 'duplicate' sequences": lambda sequence_count: [
            sequence for sequence, count in sequence_count.items() if count >= 2
        ],
        "extracting 'once' sequences": lambda sequence_count: list(
            sequence_count.keys()
        ),
    }

    results = {}

    try:
        logger.info(msg="/ Task: Count sequence frequency")
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")
        logger.log_with_borders(
            level=logging.INFO,
            message=f"Processing sequences for dataset '{dataset_name}'.",
            border="|",
            length=100,
        )

        # Count occurrences of each sequence
        sequence_count = count_sequence_occurrences(df=df, logger=logger)
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

        # Execute each step in sequence processing
        for step_name, step_func in steps.items():
            logger.log_with_borders(
                level=logging.INFO,
                message=f"Starting to {step_name} from '{dataset_name}'.",
                border="|",
                length=100,
            )
            results[step_name.split()[1].strip("'")] = step_func(sequence_count)
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

        # Log results summary
        logger.log_with_borders(
            level=logging.INFO,
            message=f"Sequence processing completed for '{dataset_name}'.\n"
            f"Unique: {len(results['unique'])}, "
            f"Duplicate: {len(results['duplicate'])}, "
            f"Once: {len(results['once'])}.",
            border="|",
            length=100,
        )
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

        return results

    except Exception:
        logger.exception(msg="Unexpected error in 'count_sequences()'.")
        raise


def write_fasta_file(df: pd.DataFrame, output_fasta: str, logger: CustomLogger) -> None:
    """
    Write sequence data to a FASTA file.

    Each sequence is written with an identifier and its corresponding sequence.
    The identifier is stripped of leading/trailing spaces and all spaces are replaced with underscores `_`.

    The sequence itself is also stripped of spaces to ensure correct FASTA format.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame containing 'ID' and 'Sequence' columns.
    output_fasta : str
        The file path to save the FASTA file.
    logger : CustomLogger
        Logger instance for logging the process.

    Returns
    -------
    None
    """
    try:
        # Log the start of the FASTA writing process
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")
        logger.log_with_borders(
            level=logging.INFO,
            message="Starting FASTA file writing.",
            border="|",
            length=100,
        )

        # Write to FASTA file
        with open(output_fasta, "w", encoding="utf-8") as f:
            for _, row in df.iterrows():
                sequence = row["Sequence"].strip().replace(" ", "")
                identifier = row["ID"].strip().replace(" ", "_")
                f.write(f">{identifier}\n{sequence}\n")

        # Log success message
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")
        logger.log_with_borders(
            level=logging.INFO,
            message=f"Saved:\n'{output_fasta}'",
            border="|",
            length=100,
        )

    except Exception:
        logger.exception(msg="Unexpected error in 'write_fasta_file()'.")
        raise


def split_and_save_dataset(
    df: pd.DataFrame, output_dir: str, test_size: float, logger: CustomLogger
):
    """
    Splits the dataset into training and testing sets and saves them as CSV and FASTA files.

    The splitting process ensures class balance using stratified sampling based on the 'Targets' column.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to be split.
    output_dir : str
        The directory where the split datasets will be saved.
    test_size : float
        The proportion of the dataset to be used as the test set.
    logger : CustomLogger
        Logger instance for logging the process.

    Returns
    -------
    None
    """
    try:
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Log the start of dataset splitting
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")
        logger.log_with_borders(
            level=logging.INFO,
            message=f"Splitting dataset into {int((1 - test_size) * 100)}% train / {int(test_size * 100)}% test.",
            border="|",
            length=100,
        )

        # Create bins for Log MIC Value to enable stratified sampling
        log_mic_bins = pd.cut(df["Log MIC Value"], bins=10, labels=False)

        # Split the dataset into train and test sets with stratification on Log MIC Value
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=42,
            stratify=log_mic_bins,  # Stratify based on Log MIC Value bins
        )

        # Define file paths for saving train and test datasets
        train_output_csv_file = os.path.join(output_dir, "train.csv")
        train_output_fasta_file = os.path.join(output_dir, "train.fasta")
        test_output_csv_file = os.path.join(output_dir, "test.csv")
        test_output_fasta_file = os.path.join(output_dir, "test.fasta")

        # Save train and test datasets to CSV files
        train_df.to_csv(path_or_buf=train_output_csv_file, index=False)
        test_df.to_csv(path_or_buf=test_output_csv_file, index=False)

        # Log completion message with file paths
        logger.log_with_borders(
            level=logging.INFO,
            message=f"Saved:\nTrain: ({train_df.shape[0]}) rows saved to '{train_output_csv_file}'\nTest: ({test_df.shape[0]}) rows saved to '{test_output_csv_file}'",
            border="|",
            length=100,
        )

        # Save train and test datasets as FASTA files
        write_fasta_file(
            df=train_df, output_fasta=train_output_fasta_file, logger=logger
        )
        write_fasta_file(df=test_df, output_fasta=test_output_fasta_file, logger=logger)
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

    except Exception:
        logger.exception(msg="Unexpected error in 'split_and_save_dataset()'.")
        raise


# ============================== Pipeline Entry Point ==============================
def run_clean_pipeline(base_path: str, logger: CustomLogger) -> None:
    """

    Parameters
    ----------
    base_path : str
        Absolute base path of the project (used to construct file paths).
    logger : CustomLogger
        Logger for structured logging throughout the pipeline.

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

        # Loop over each strain type
        for strain_index, (strain, suffix) in enumerate(strains.items(), start=1):

            # Logging: strain section
            logger.info(msg=f"/ {strain_index}")
            logger.add_divider(level=logging.INFO, length=40, border="+", fill="-")
            logger.log_with_borders(
                level=logging.INFO,
                message=f"Strain - '{strain}'",
                border="|",
                length=40,
            )
            logger.add_divider(level=logging.INFO, length=40, border="+", fill="-")
            logger.add_spacer(level=logging.INFO, lines=1)

            # Load data for the target strain
            data_input_file = os.path.join(
                base_path, f"data/interim/{suffix}/All_Integrated.csv"
            )
            data = pd.read_csv(filepath_or_buffer=data_input_file)

            # Filter out invalid values (-inf, inf, NaN) in the "Log MIC Value" column
            data = data[np.isfinite(data["Log MIC Value"])]

            # Count sequence frequencies for the current dataset

            _ = count_sequences(
                df=data,
                dataset_name=f"{suffix}/All_Integrated.csv",
                logger=logger,
            )
            logger.add_spacer()

            # Calculate Z-score for the "Log MIC Value" column

            logger.info(msg="/ Task: Calculating Z-score for the 'Log MIC Value'.")
            logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")
            data["Log MIC Z-score"] = zscore(data["Log MIC Value"])

            # Filter rows where Z-score is between -3 and 3, and drop the Z-score column
            logger.log_with_borders(
                level=logging.INFO,
                message="Retaining rows where Z-score is between -3 and 3.",
                border="|",
                length=100,
            )
            filtered_data = data[
                (data["Log MIC Z-score"] >= -3) & (data["Log MIC Z-score"] <= 3)
            ].drop(columns=["Log MIC Z-score"])

            # Save the filtered data
            logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")
            filtered_data_output_file = os.path.join(
                base_path,
                f"data/interim/{suffix}/All_Integrated_filter_by_zscore.csv",
            )
            filtered_data.to_csv(
                path_or_buf=filtered_data_output_file,
                index=False,
            )
            logger.log_with_borders(
                level=logging.INFO,
                message=f"Saved:\n'{filtered_data_output_file}'",
                border="|",
                length=100,
            )
            logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")
            logger.add_spacer()

            # Reload the filtered data
            data_input_file = os.path.join(
                base_path,
                f"data/interim/{suffix}/All_Integrated_filter_by_zscore.csv",
            )
            data = pd.read_csv(filepath_or_buffer=data_input_file)

            # Recount sequence frequencies after filtering
            _ = count_sequences(
                df=data,
                dataset_name=f"{suffix}/All_Integrated_filter_by_zscore.csv",
                logger=logger,
            )
            logger.add_spacer()

            # Aggregate data based on min of "Log MIC Value"
            logger.info(msg="/ Task: Aggregate data based on 'Min'.")
            logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")
            aggregated_data = (
                data.groupby("Sequence")
                .agg(
                    Log_MIC_Value_Min=("Log MIC Value", "min"),
                )
                .reset_index()
            )

            # Rename aggregated columns for better readability
            aggregated_data = aggregated_data.rename(
                columns={
                    "Log_MIC_Value_Min": "Log MIC Value Min",
                }
            )

            # Remove duplicate sequences and unwanted columns
            data_unique = data.drop_duplicates(subset=["Sequence"], keep="first")
            columns_to_remove = ["Log MIC Value", "MIC", "MIC Value", "Unit"]
            data_cleaned = data_unique.drop(columns=columns_to_remove, errors="ignore")

            # Merge cleaned data with aggregated data
            data_with_aggregated = pd.merge(
                aggregated_data, data_cleaned, on="Sequence", how="left"
            )

            # Update "Targets" column with the strain name
            data_with_aggregated["Targets"] = strain

            # Reorder columns for better organization
            cols = [
                "ID",
                "Sequence",
                "Targets",
                "Sequence Length",
                "Molecular Weight",
                "Log MIC Value Min",
            ]
            data_with_aggregated = data_with_aggregated[cols]

            # Rename aggregated columns for better readability
            data_with_aggregated = data_with_aggregated.rename(
                columns={
                    "Log MIC Value Min": "Log MIC Value",
                }
            )

            # Save the aggregated data
            data_with_aggregated_output_dir = os.path.join(
                base_path,
                f"data/processed/{suffix}/",
            )
            os.makedirs(name=data_with_aggregated_output_dir, exist_ok=True)
            data_with_aggregated_output_file = os.path.join(
                base_path,
                f"data/processed/{suffix}/All_Integrated_aggregated.csv",
            )
            data_with_aggregated.to_csv(
                path_or_buf=data_with_aggregated_output_file,
                index=False,
            )
            logger.log_with_borders(
                level=logging.INFO,
                message=f"Saved:\n'{data_with_aggregated_output_file}'",
                border="|",
                length=100,
            )
            logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")
            logger.add_spacer()

            # Reload the aggregated data
            data_input_file = os.path.join(
                base_path,
                f"data/processed/{suffix}/All_Integrated_aggregated.csv",
            )
            data = pd.read_csv(filepath_or_buffer=data_input_file)

            # Count sequence frequencies for the aggregated dataset
            _ = count_sequences(
                df=data,
                dataset_name=f"{suffix}/All_Integrated_aggregated.csv",
                logger=logger,
            )
            logger.add_spacer()

            # Write sequences to FASTA file
            logger.info(msg="/ Task: Writing sequences to FASTA file")
            data_output_fasta_file = os.path.join(
                base_path,
                f"data/processed/{suffix}/All_Integrated_aggregated.fasta",
            )
            write_fasta_file(
                df=data, output_fasta=data_output_fasta_file, logger=logger
            )
            logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")
            logger.add_spacer()

            # Split and save dataset for training and testing
            logger.info(msg="/ Task: Splitting dataset for training and testing.")
            split_and_save_dataset(
                df=data,
                output_dir=data_with_aggregated_output_dir,
                test_size=0.2,  # 80:20 train-test split
                logger=logger,
            )
            logger.add_spacer()

    except Exception:
        logger.exception(msg="Unexpected error in 'run_collect_pipeline()'.")
        raise
