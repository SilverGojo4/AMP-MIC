# pylint: disable=line-too-long, import-error, wrong-import-position
"""
This module handles iFeature-based feature extraction for antimicrobial peptide (AMP) sequences.
It supports various descriptor types including AAC, PAAC, CTDD, and GAAC. The extracted features
are automatically converted and merged into the corresponding train/test datasets for each strain.

The processing pipeline:
1. Executes iFeature scripts via subprocess for each descriptor.
2. Converts the generated TSV output into CSV format with prefixed column names.
3. Merges the extracted features into the original dataset by sequence ID.
4. Repeats the process for all combinations of strains and data splits (train/test).

The core functions in this module are:
- `ifeature_aac()`: Runs AAC (Amino Acid Composition) feature extraction.
- `ifeature_paac()`: Runs PAAC (Pseudo Amino Acid Composition) feature extraction with a configurable lambda.
- `ifeature_ctdd()`: Runs CTDD (Composition, Transition, Distribution Descriptor) feature extraction.
- `ifeature_gaac()`: Runs GAAC (Grouped Amino Acid Composition) feature extraction.
- `convert_tsv_to_csv()`: Converts iFeature TSV output to CSV with proper column renaming.
- `merge_csv_by_id()`: Merges feature CSV into the target dataset using 'ID' as key.
- `run_ifeature_pipeline()`: Entry point that orchestrates the full feature extraction process.

The `run_ifeature_pipeline()` function serves as the entry point for running the entire feature
extraction workflow. It applies all supported encodings across all processed strain datasets
for both training and testing.
"""
# ============================== Standard Library Imports ==============================
import logging
import os
import subprocess
import sys

# ============================== Third-Party Library Imports ==============================
import pandas as pd

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
def convert_tsv_to_csv(
    input_tsv: str,
    output_csv: str,
    feature_prefix: str,
    logger: CustomLogger,
) -> None:
    """
    Convert a TSV file to a CSV file and rename feature columns by adding a specified prefix.

    Parameters
    ----------
    input_tsv : str
        Path to the input TSV file.
    output_csv : str
        Path to the output CSV file.
    feature_prefix : str
        Prefix to be added before each feature column name.
    logger : CustomLogger
        Logger for both general and error logging.

    Returns
    -------
    None
    """
    try:
        # Log the start of conversion
        logger.log_with_borders(
            level=logging.INFO,
            message="Starting conversion of TSV to CSV.",
            border="|",
            length=100,
        )

        # Load the TSV file into a DataFrame
        df = pd.read_csv(input_tsv, sep="\t")

        # Check if the file is empty
        if df.empty:
            raise ValueError(f"Input file '{input_tsv}' is empty.")

        # Ensure the first column is an identifier column (ID)
        if df.columns[0] != "#":
            raise KeyError(
                "The first column in the TSV file must be an identifier column (expected '#')."
            )

        # Rename the first column from '#' to 'ID'
        df.rename(columns={"#": "ID"}, inplace=True)

        # Rename feature columns (exclude the first column, which is now 'ID')
        df.rename(
            columns={col: f"{feature_prefix}|{col}" for col in df.columns[1:]},
            inplace=True,
        )

        # Save as CSV file
        df.to_csv(output_csv, index=False)

        # Log the successful completion of conversion
        logger.log_with_borders(
            level=logging.INFO,
            message=f"Successfully converted with feature prefix '{feature_prefix}'.",
            border="|",
            length=100,
        )
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")
        logger.log_with_borders(
            level=logging.INFO,
            message=f"Saved CSV file:\n'{output_csv}'",
            border="|",
            length=100,
        )
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

        # Delete the TSV file after successful conversion
        os.remove(input_tsv)
        logger.log_with_borders(
            level=logging.INFO,
            message=f"Deleted temporary TSV file:\n'{input_tsv}'",
            border="|",
            length=100,
        )

    except FileNotFoundError:
        logger.error(
            msg=f"FileNotFoundError in 'convert_tsv_to_csv': The file '{input_tsv}' was not found."
        )
        raise

    except KeyError as e:
        logger.error(msg=f"KeyError in 'convert_tsv_to_csv': {str(e)}")
        raise

    except ValueError as e:
        logger.error(msg=f"ValueError in 'convert_tsv_to_csv': {str(e)}")
        raise

    except Exception as e:
        logger.error(msg=f"Unexpected error in 'convert_tsv_to_csv': {str(e)}")
        raise


def merge_csv_by_id(
    file1: str,
    file2: str,
    logger: CustomLogger,
) -> None:
    """
    Merge two CSV files based on the 'ID' column and overwrite the first file.

    Parameters
    ----------
    file1 : str
        Path to the first CSV file (main data).
    file2 : str
        Path to the second CSV file (feature data).
    logger : CustomLogger
        Logger instance for recording the process.

    Returns
    -------
    None
    """
    try:
        # Log the start of merging
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")
        logger.log_with_borders(
            level=logging.INFO,
            message="Starting merge of based on 'ID' column.",
            border="|",
            length=100,
        )

        # Load the first CSV file (main data)
        df1 = pd.read_csv(file1)

        # Load the second CSV file (feature data)
        df2 = pd.read_csv(file2)

        # Check if the files are empty
        if df1.empty:
            raise ValueError(f"Error: '{file1}' is empty.")
        if df2.empty:
            raise ValueError(f"Error: '{file2}' is empty.")

        # Ensure both files contain the 'ID' column
        if "ID" not in df1.columns:
            raise KeyError(f"Missing 'ID' column in '{file1}'.")
        if "ID" not in df2.columns:
            raise KeyError(f"Missing 'ID' column in '{file2}'.")

        # Merge the two dataframes based on the 'ID' column (left join)
        merged_df = df1.merge(df2, on="ID", how="left")

        # Save the merged dataframe back to file1 (overwrite)
        merged_df.to_csv(file1, index=False)

        # Log the successful merge
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")
        logger.log_with_borders(
            level=logging.INFO,
            message=f"Successfully merged '{file2}' into '{file1}'.",
            border="|",
            length=100,
        )
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

        # Delete file2 after successful merge
        os.remove(file2)
        logger.log_with_borders(
            level=logging.INFO,
            message=f"Deleted temporary CSV file:\n'{file2}'",
            border="|",
            length=100,
        )
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

    except Exception:
        logger.exception(msg="Unexpected error in 'merge_csv_by_id()'.")
        raise


def ifeature_aac(
    base_path: str,
    input_fasta: str,
    output_file: str,
    logger: CustomLogger,
) -> None:
    """
    Executes iFeature AAC feature extraction via subprocess and saves the output as a TSV file.

    This function runs the iFeature script using a subprocess, verifies successful execution,
    and ensures the output file is generated correctly.

    Parameters
    ----------
    base_path : str
        The root directory of the project.
    input_fasta : str
        Path to the input FASTA file.
    output_file : str
        Path to the final output CSV file.
    logger : CustomLogger
        Logger instance for recording the process.

    Returns
    -------
    None
    """
    try:
        # Convert the user-provided CSV output path to an internal TSV path
        output_tsv = output_file.replace(".csv", ".tsv")

        # Define script path for iFeature AAC
        script_path = os.path.join(base_path, "src/utils/iFeature/iFeature.py")

        # Ensure the output directory exists
        output_dir = os.path.dirname(output_file)
        os.makedirs(output_dir, exist_ok=True)

        # Construct command for subprocess execution
        command = [
            "python",
            script_path,
            "--file",
            input_fasta,
            "--out",
            output_tsv,
            "--type",
            "AAC",  # Specify the type of feature extraction (AAC)
        ]

        # Log the start of the execution
        logger.info(msg="/ Task: Starting iFeature 'AAC' feature extraction")
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

        # Execute the command and capture output
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=600,
            check=True,
        )

        # Log standard output and error messages for debugging
        if result.stdout.strip():
            logger.log_with_borders(
                level=logging.INFO,
                message=f"iFeature 'AAC' Output:\n{result.stdout.strip()}",
                border="|",
                length=100,
            )
        if result.stderr.strip():
            logger.log_with_borders(
                level=logging.ERROR,
                message=f"iFeature 'AAC' Error Output:\n{result.stderr.strip()}",
                border="|",
                length=100,
            )

        # Check if the subprocess executed successfully
        if result.returncode != 0:
            logger.error(
                msg=f"Subprocess failed in 'ifeature_aac' with error:\n{result.stderr.strip()}"
            )
            raise subprocess.CalledProcessError(result.returncode, command)

        # Verify the existence of the output TSV file
        if not os.path.exists(output_tsv):
            logger.error(
                msg=f"iFeature 'AAC' execution completed, but output file '{output_tsv}' was not created."
            )
            raise FileNotFoundError(f"Output file '{output_tsv}' was not generated.")

        # Log successful execution
        logger.log_with_borders(
            level=logging.INFO,
            message=f"Successfully saved iFeature 'AAC' results:\n'{output_tsv}'",
            border="|",
            length=100,
        )
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

        # Convert the output TSV file to CSV
        convert_tsv_to_csv(
            input_tsv=output_tsv,
            output_csv=output_file,
            feature_prefix="AAC",
            logger=logger,
        )
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

    except Exception:
        logger.exception(msg="Unexpected error in 'ifeature_aac()'.")
        raise


def ifeature_paac(
    base_path: str,
    input_fasta: str,
    output_file: str,
    lambda_value: int,
    logger: CustomLogger,
) -> None:
    """
    Executes iFeature PAAC feature extraction via subprocess, saves output as a TSV file,
    and converts it to a CSV file (TSV will be deleted automatically).

    Parameters
    ----------
    base_path : str
        The root directory of the project.
    input_fasta : str
        Path to the input FASTA file.
    output_file : str
        Path to the final output CSV file.
    lambda_value : int
        The lambda parameter for PAAC encoding.
    logger : CustomLogger
        Logger instance for recording the process.

    Returns
    -------
    None
    """
    try:
        # Convert the user-provided CSV output path to an internal TSV path
        output_tsv = output_file.replace(".csv", ".tsv")

        # Define the script path for PAAC encoding
        script_path = os.path.join(base_path, "src/utils/iFeature/codes/PAAC.py")

        # Ensure the output directory exists
        output_dir = os.path.dirname(output_file)
        os.makedirs(output_dir, exist_ok=True)

        # Construct the command for subprocess execution
        command = [
            "python",
            script_path,
            input_fasta,
            str(lambda_value),  # Lambda value parameter
            output_tsv,  # Internal TSV file
        ]

        # Log the start of the execution
        logger.info(
            msg=f"/ Task: Starting iFeature 'PAAC' feature extraction (Lambda = {lambda_value})"
        )
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

        # Execute the command and capture output
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=600,
            check=True,
        )

        # Log standard output and error messages for debugging
        if result.stdout.strip():
            logger.log_with_borders(
                level=logging.INFO,
                message=f"iFeature 'PAAC' Output:\n{result.stdout.strip()}",
                border="|",
                length=100,
            )
        if result.stderr.strip():
            logger.log_with_borders(
                level=logging.ERROR,
                message=f"iFeature 'PAAC' Error Output:\n{result.stderr.strip()}",
                border="|",
                length=100,
            )

        # Check if the subprocess executed successfully
        if result.returncode != 0:
            logger.error(
                msg=f"Subprocess failed in 'ifeature_paac' with error:\n{result.stderr.strip()}"
            )
            raise subprocess.CalledProcessError(result.returncode, command)

        # Verify the existence of the output TSV file
        if not os.path.exists(output_tsv):
            logger.error(
                msg=f"iFeature 'PAAC' execution completed, but output file '{output_tsv}' was not created."
            )
            raise FileNotFoundError(f"Output file '{output_tsv}' was not generated.")

        # Log successful execution
        logger.log_with_borders(
            level=logging.INFO,
            message=f"Successfully saved iFeature 'PAAC' results:\n'{output_tsv}'",
            border="|",
            length=100,
        )
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

        # Convert the output TSV file to CSV (TSV will be deleted automatically)
        convert_tsv_to_csv(
            input_tsv=output_tsv,
            output_csv=output_file,
            feature_prefix="PAAC",
            logger=logger,
        )
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

    except Exception:
        logger.exception(msg="Unexpected error in 'ifeature_paac()'.")
        raise


def ifeature_ctdd(
    base_path: str,
    input_fasta: str,
    output_file: str,
    logger: CustomLogger,
) -> None:
    """
    Executes iFeature CTDD feature extraction via subprocess, saves output as a TSV file,
    and converts it to a CSV file (TSV will be deleted automatically).

    Parameters
    ----------
    base_path : str
        The root directory of the project.
    input_fasta : str
        Path to the input FASTA file.
    output_file : str
        Path to the final output CSV file.
    logger : CustomLogger
        Logger instance for recording the process.

    Returns
    -------
    None
    """
    try:
        # Convert the user-provided CSV output path to an internal TSV path
        output_tsv = output_file.replace(".csv", ".tsv")

        # Define the script path for iFeature CTDD execution
        script_path = os.path.join(base_path, "src/utils/iFeature/iFeature.py")

        # Ensure the output directory exists
        output_dir = os.path.dirname(output_file)
        os.makedirs(output_dir, exist_ok=True)

        # Construct the command for subprocess execution
        command = [
            "python",
            script_path,
            "--file",
            input_fasta,
            "--out",
            output_tsv,
            "--type",
            "CTDD",  # Specify the feature extraction type (CTDD)
        ]

        # Log the start of the execution
        logger.info(msg="/ Task: Starting iFeature 'CTDD' feature extraction")
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

        # Execute the command and capture output
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=600,
            check=True,
        )

        # Log standard output and error messages for debugging
        if result.stdout.strip():
            logger.log_with_borders(
                level=logging.INFO,
                message=f"iFeature 'CTDD' Output:\n{result.stdout.strip()}",
                border="|",
                length=100,
            )
        if result.stderr.strip():
            logger.log_with_borders(
                level=logging.ERROR,
                message=f"iFeature 'CTDD' Error Output:\n{result.stderr.strip()}",
                border="|",
                length=100,
            )

        # Check if the subprocess executed successfully
        if result.returncode != 0:
            logger.error(
                msg=f"Subprocess failed in 'ifeature_ctdd' with error:\n{result.stderr.strip()}"
            )
            raise subprocess.CalledProcessError(result.returncode, command)

        # Verify the existence of the output TSV file
        if not os.path.exists(output_tsv):
            logger.error(
                msg=f"iFeature 'CTDD' execution completed, but output file '{output_tsv}' was not created."
            )
            raise FileNotFoundError(f"Output file '{output_tsv}' was not generated.")

        # Log successful execution
        logger.log_with_borders(
            level=logging.INFO,
            message=f"Successfully saved iFeature 'CTDD' results:\n'{output_tsv}'",
            border="|",
            length=100,
        )
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

        # Convert the output TSV file to CSV (TSV will be deleted automatically)
        convert_tsv_to_csv(
            input_tsv=output_tsv,
            output_csv=output_file,
            feature_prefix="CTDD",
            logger=logger,
        )
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

    except Exception:
        logger.exception(msg="Unexpected error in 'ifeature_ctdd()'.")
        raise


def ifeature_gaac(
    base_path: str,
    input_fasta: str,
    output_file: str,
    logger: CustomLogger,
) -> None:
    """
    Executes iFeature GAAC feature extraction via subprocess, saves output as a TSV file,
    and converts it to a CSV file (TSV will be deleted automatically).

    Parameters
    ----------
    base_path : str
        The root directory of the project.
    input_fasta : str
        Path to the input FASTA file.
    output_file : str
        Path to the final output CSV file.
    logger : CustomLogger
        Logger instance for recording the process.

    Returns
    -------
    None
    """
    try:
        # Convert the user-provided CSV output path to an internal TSV path
        output_tsv = output_file.replace(".csv", ".tsv")

        # Define the script path for iFeature GAAC execution
        script_path = os.path.join(base_path, "src/utils/iFeature/iFeature.py")

        # Ensure the output directory exists
        output_dir = os.path.dirname(output_file)
        os.makedirs(output_dir, exist_ok=True)

        # Construct the command for subprocess execution
        command = [
            "python",
            script_path,
            "--file",
            input_fasta,
            "--out",
            output_tsv,
            "--type",
            "GAAC",  # Specify the feature extraction type (GAAC)
        ]

        # Log the start of the execution
        logger.info(msg="/ Task: Starting iFeature 'GAAC' feature extraction")
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

        # Execute the command and capture output
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=600,
            check=True,
        )

        # Log standard output and error messages for debugging
        if result.stdout.strip():
            logger.log_with_borders(
                level=logging.INFO,
                message=f"iFeature 'GAAC' Output:\n{result.stdout.strip()}",
                border="|",
                length=100,
            )
        if result.stderr.strip():
            logger.log_with_borders(
                level=logging.ERROR,
                message=f"iFeature 'GAAC' Error Output:\n{result.stderr.strip()}",
                border="|",
                length=100,
            )

        # Check if the subprocess executed successfully
        if result.returncode != 0:
            logger.error(
                msg=f"Subprocess failed in 'ifeature_gaac' with error:\n{result.stderr.strip()}"
            )
            raise subprocess.CalledProcessError(result.returncode, command)

        # Verify the existence of the output TSV file
        if not os.path.exists(output_tsv):
            logger.error(
                msg=f"iFeature 'GAAC' execution completed, but output file '{output_tsv}' was not created."
            )
            raise FileNotFoundError(f"Output file '{output_tsv}' was not generated.")

        # Log successful execution
        logger.log_with_borders(
            level=logging.INFO,
            message=f"Successfully saved iFeature 'GAAC' results:\n'{output_tsv}'",
            border="|",
            length=100,
        )
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

        # Convert the output TSV file to CSV (TSV will be deleted automatically)
        convert_tsv_to_csv(
            input_tsv=output_tsv,
            output_csv=output_file,
            feature_prefix="GAAC",
            logger=logger,
        )
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

    except Exception:
        logger.exception(msg="Unexpected error in 'ifeature_gaac()'.")
        raise


# ============================== Pipeline Entry Point ==============================
def run_ifeature_pipeline(base_path: str, logger: CustomLogger) -> None:
    """
    Entry point for the iFeature-based feature extraction pipeline.

    This function performs feature extraction on AMP sequences (train/test) across multiple strains
    using various iFeature descriptors including AAC, PAAC, CTDD, and GAAC. It iterates over each
    strain and data split, runs the feature encoders, and merges the resulting feature vectors
    into the respective CSV datasets for downstream modeling.

    For each (strain, dataset type) pair, the following steps are executed:
    1. Extract AAC (Amino Acid Composition) features from the FASTA sequences.
    2. Extract PAAC (Pseudo Amino Acid Composition) features with lambda = 5.
    3. Extract CTDD (Composition, Transition, Distribution Descriptor) features.
    4. Extract GAAC (Grouped Amino Acid Composition) features.
    5. Merge each set of extracted features into the corresponding training/test CSV file by ID.

    Parameters
    ----------
    base_path : str
        Absolute base path of the project (used to construct file paths).
    logger : CustomLogger
        Logger instance for structured logging throughout the pipeline.

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

        # Types of data splits to process (training and testing)
        data_type = ["train", "test"]

        # Loop over each strain type
        for strain_index, (strain, suffix) in enumerate(strains.items(), start=1):

            for data_index, data in enumerate(data_type, start=1):

                # Logging: strain section
                logger.info(msg=f"/ {strain_index}.{data_index}")
                logger.add_divider(level=logging.INFO, length=60, border="+", fill="-")
                logger.log_with_borders(
                    level=logging.INFO,
                    message=f"Strain - '{strain}' -> Data - '{data}'",
                    border="|",
                    length=60,
                )
                logger.add_divider(level=logging.INFO, length=60, border="+", fill="-")
                logger.add_spacer(level=logging.INFO, lines=1)

                # Define input and output paths
                input_fasta_file = os.path.join(
                    base_path, f"data/processed/{suffix}/{data}.fasta"
                )
                output_file = os.path.join(
                    base_path, f"data/processed/{suffix}/{data}.csv"
                )
                output_aac_file = os.path.join(
                    base_path, f"data/processed/{suffix}/{data}_aac.csv"
                )
                output_paac_file = os.path.join(
                    base_path, f"data/processed/{suffix}/{data}_paac.csv"
                )
                output_ctdd_file = os.path.join(
                    base_path, f"data/processed/{suffix}/{data}_ctdd.csv"
                )
                output_gaac_file = os.path.join(
                    base_path, f"data/processed/{suffix}/{data}_gaac.csv"
                )

                # Perform AAC feature extraction
                ifeature_aac(
                    base_path=base_path,
                    input_fasta=input_fasta_file,
                    output_file=output_aac_file,
                    logger=logger,
                )

                # Insert a blank line in the log for readability
                logger.add_spacer(level=logging.INFO, lines=1)

                # Merge the AAC features into the training dataset
                logger.info(msg="/ Task: Merging 'AAC' features")
                merge_csv_by_id(
                    file1=output_file,
                    file2=output_aac_file,
                    logger=logger,
                )

                # Insert a blank line in the log for readability
                logger.add_spacer(level=logging.INFO, lines=1)

                # Perform PAAC feature extraction
                ifeature_paac(
                    base_path=base_path,
                    input_fasta=input_fasta_file,
                    output_file=output_paac_file,
                    lambda_value=5,
                    logger=logger,
                )

                # Insert a blank line in the log for readability
                logger.add_spacer(level=logging.INFO, lines=1)

                # Merge the PAAC features into the training dataset
                logger.info(msg="/ Task: Merging 'PAAC' features")
                merge_csv_by_id(
                    file1=output_file,
                    file2=output_paac_file,
                    logger=logger,
                )

                # Insert a blank line in the log for readability
                logger.add_spacer(level=logging.INFO, lines=1)

                # Perform CTDD feature extraction
                ifeature_ctdd(
                    base_path=base_path,
                    input_fasta=input_fasta_file,
                    output_file=output_ctdd_file,
                    logger=logger,
                )

                # Insert a blank line in the log for readability
                logger.add_spacer(level=logging.INFO, lines=1)

                # Merge the CTDD features into the training dataset
                logger.info(msg="/ Task: Merging 'CTDD' features")
                merge_csv_by_id(
                    file1=output_file,
                    file2=output_ctdd_file,
                    logger=logger,
                )

                # Insert a blank line in the log for readability
                logger.add_spacer(level=logging.INFO, lines=1)

                # Perform GAAC feature extraction
                ifeature_gaac(
                    base_path=base_path,
                    input_fasta=input_fasta_file,
                    output_file=output_gaac_file,
                    logger=logger,
                )

                # Insert a blank line in the log for readability
                logger.add_spacer(level=logging.INFO, lines=1)

                # Merge the GAAC features into the training dataset
                logger.info(msg="/ Task: Merging 'GAAC' features")
                merge_csv_by_id(
                    file1=output_file,
                    file2=output_gaac_file,
                    logger=logger,
                )

                # Insert a blank line in the log for readability
                logger.add_spacer(level=logging.INFO, lines=1)

    except Exception:
        logger.exception(msg="Unexpected error in 'run_ifeature_pipeline()'.")
        raise
