# pylint: disable=line-too-long, import-error, wrong-import-position, singleton-comparison, too-many-locals, too-many-nested-blocks, too-many-branches, too-many-statements, too-many-lines
"""
This module handles the collection and preprocessing of antimicrobial peptide (AMP) data
from multiple public databases, including DBAASP, dbAMP, and DRAMP. The module provides
functions for data filtering, sequence length calculation, molecular weight computation,
and concentration transformations, with structured logging for progress tracking and error handling.

The processing pipeline:
1. Filters sequences containing only natural amino acids.
2. Calculates sequence lengths, molecular weights, and MIC values.
3. Filters sequences based on length.
4. Saves the processed data filtered by strain for downstream analysis.

The core functions in this module are:
- `filter_natural_aa()`: Filters out sequences that contain non-natural amino acids.
- `calculate_sequence_length()`: Calculates and adds a sequence length column to the DataFrame.
- `sequence_length_cut()`: Filters sequences by length, retaining only those within the specified bounds.
- `calculate_molecular_weight()`: Calculates the molecular weight for each sequence.
- `process_sequences()`: Executes all sequence processing steps in order.
- `calculate_concentration()`: Extracts and calculates the concentration value from a string.
- `transform_concentration()`: Transforms the concentration values based on units and calculates their logarithmic values.
- `filter_save_by_target_strains()`: Filters the data by strain, saves the results to a CSV, and logs the process.

The `run_collect_pipeline()` function serves as the entry point for running the entire collection
pipeline. It organizes the data extraction and preprocessing tasks from the different databases and
manages logging for each step of the process.
"""
# ============================== Standard Library Imports ==============================
import logging
import math
import os
import re
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
def filter_natural_aa(
    df: pd.DataFrame,
    logger: CustomLogger,
) -> pd.DataFrame:
    """
    Filter out sequences that do not contain only natural amino acids.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame containing biological sequences.
    logger : CustomLogger
        Logger instance for tracking progress and errors.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing only sequences with natural amino acids.
    """
    # Define the set of natural amino acids
    natural_aa_set = set("RHKDESTNQCGPAILMFWYV")

    # Initialize the 'Nature' column to indicate whether a sequence contains only natural amino acids
    df["Nature"] = False

    try:
        # -------------------- Logging: Start --------------------
        logger.log_with_borders(
            level=logging.INFO,
            message="Starting to filter sequences containing only natural amino acids.",
            border="|",
            length=100,
        )

        # Check each sequence to ensure it contains only natural amino acids
        df["Nature"] = df["Sequence"].apply(
            lambda seq: all(char in natural_aa_set for char in str(seq))
        )

        # Retain only sequences that passed the filter
        filtered_df = df[df["Nature"] == True].copy()

        # -------------------- Logging: Completion --------------------
        logger.log_with_borders(
            level=logging.INFO,
            message=f"Completed filtering sequences. {len(filtered_df)} sequences remaining.",
            border="|",
            length=100,
        )

        return filtered_df

    except Exception:
        logger.exception(msg="Unexpected error in 'filter_natural_aa()'.")
        raise


def calculate_sequence_length(
    df: pd.DataFrame,
    logger: CustomLogger,
) -> pd.DataFrame:
    """
    Calculate the length of each sequence in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame containing the input sequences.
    logger : CustomLogger
        Logger instance for tracking progress and errors.

    Returns
    -------
    pd.DataFrame
        A DataFrame with an updated column containing sequence lengths.
    """
    try:
        # -------------------- Logging: Start --------------------
        logger.log_with_borders(
            level=logging.INFO,
            message="Starting to calculate sequence lengths.",
            border="|",
            length=100,
        )

        # Calculate sequence lengths and add a new column
        df["Sequence Length"] = df["Sequence"].apply(len)

        # -------------------- Logging: Completion --------------------
        logger.log_with_borders(
            level=logging.INFO,
            message="Completed calculating sequence lengths.",
            border="|",
            length=100,
        )

        return df

    except Exception:
        logger.exception(msg="Unexpected error in 'calculate_sequence_length()'.")
        raise


def sequence_length_cut(
    df: pd.DataFrame,
    lower_bound: int,
    upper_bound: int,
    logger: CustomLogger,
) -> pd.DataFrame:
    """
    Filter sequences by length, retaining only sequences with lengths between the lower and upper bounds.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame containing the input sequences.
    lower_bound : int
        The lower bound for sequence length filtering.
    upper_bound : int
        The upper bound for sequence length filtering.
    logger : CustomLogger
        Logger instance for tracking progress and errors.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing only sequences with lengths between the lower and upper bounds.
    """
    try:
        # -------------------- Logging: Start --------------------
        logger.log_with_borders(
            level=logging.INFO,
            message=f"Starting to filter sequences with lengths between {lower_bound} and {upper_bound}.",
            border="|",
            length=100,
        )

        # Apply the length filter and create a new DataFrame
        filtered_df = df[
            (df["Sequence Length"] > lower_bound)
            & (df["Sequence Length"] < upper_bound)
        ].copy()

        # -------------------- Logging: Completion --------------------
        logger.log_with_borders(
            level=logging.INFO,
            message=f"Completed filtering sequences by length. {len(filtered_df)} sequences retained.",
            border="|",
            length=100,
        )

        return filtered_df

    except Exception:
        logger.exception(msg="Unexpected error in 'sequence_length_cut()'.")
        raise


def calculate_molecular_weight(
    df: pd.DataFrame,
    logger: CustomLogger,
) -> pd.DataFrame:
    """
    Calculate the molecular weight of sequences.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame containing the input sequences.
    logger : CustomLogger
        Logger instance for tracking progress and errors.

    Returns
    -------
    pd.DataFrame
        A DataFrame updated with the molecular weights.
    """
    # Define the atomic weight of each amino acid
    weights = {
        "A": 71.08,
        "C": 103.15,
        "D": 115.09,
        "E": 129.12,
        "F": 147.18,
        "G": 57.05,
        "H": 137.14,
        "I": 113.16,
        "K": 128.18,
        "L": 113.16,
        "M": 131.20,
        "N": 114.11,
        "P": 97.12,
        "Q": 128.13,
        "R": 156.19,
        "S": 87.08,
        "T": 101.11,
        "V": 99.13,
        "W": 186.22,
        "Y": 163.18,
    }

    try:
        # -------------------- Logging: Start --------------------
        logger.log_with_borders(
            level=logging.INFO,
            message="Starting to calculate molecular weights for sequences.",
            border="|",
            length=100,
        )

        # Calculate molecular weights for each sequence and add a new column
        df["Molecular Weight"] = df["Sequence"].apply(
            lambda seq: sum(weights.get(aa, 0) for aa in seq)
        )

        # -------------------- Logging: Completion --------------------
        logger.log_with_borders(
            level=logging.INFO,
            message="Completed calculating molecular weights for sequences.",
            border="|",
            length=100,
        )

        return df

    except Exception:
        logger.exception(msg="Unexpected error in 'calculate_molecular_weight()'.")
        raise


def process_sequences(
    df: pd.DataFrame,
    lower_bound: int,
    upper_bound: int,
    logger: CustomLogger,
) -> pd.DataFrame:
    """
    Process the sequences by filtering out non-natural amino acids,
    calculating sequence lengths, filtering by length, and calculating molecular weight.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame containing the original sequences.
    lower_bound : int
        The lower bound for sequence length filtering.
    upper_bound : int
        The upper bound for sequence length filtering.
    logger : CustomLogger
        Logger instance for tracking progress and errors.

    Returns
    -------
    pd.DataFrame
        The processed DataFrame after applying all filters and calculations.
    """
    steps = {
        "Step 1: filtering natural amino acids": lambda df: filter_natural_aa(
            df, logger
        ),
        "Step 2: calculating sequence lengths": lambda df: calculate_sequence_length(
            df, logger
        ),
        "Step 3: filtering sequences by length": lambda df: sequence_length_cut(
            df, lower_bound, upper_bound, logger
        ),
        "Step 4: calculating molecular weights": lambda df: calculate_molecular_weight(
            df, logger
        ),
    }

    try:
        # Ensure required columns exist
        required_columns = ["Sequence"]
        for col in required_columns:
            if col not in df.columns:
                raise KeyError(f"Missing required column: {col}")

        # Make a copy of the input DataFrame to work on
        logger.info(msg="/ Task: Process the sequences.")
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")
        processed_df = df.copy()

        for _, step_func in steps.items():
            processed_df = step_func(processed_df)
            logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

        # Return the fully processed DataFrame
        return processed_df

    except Exception:
        logger.exception(msg="Unexpected error in 'process_sequences()'.")
        raise


def calculate_concentration(
    con: str,
    logger: CustomLogger,
) -> float:
    """
    Calculate the concentration value based on the given string.

    Parameters
    ----------
    con : str
        The string containing concentration information.
    logger : CustomLogger
        Logger instance for tracking progress and errors.

    Returns
    -------
    float
        The calculated concentration value.
    """
    try:
        # Handle various formats of the concentration string and extract the numeric value
        if "-" in con:
            # Case: MIC value given as a range (e.g., "10-20"), extract the first numeric value
            con_value = float(re.findall(r"-?\d+\.?\d*", con)[0])

        elif "±" in con:
            # Case: Range value specified with ± symbol
            a = float(re.findall(r"-?\d+\.?\d*", con)[0])  # First numeric value
            b = float(re.findall(r"-?\d+\.?\d*", con)[1])  # Second numeric value
            con_value = round(a - b, 3)  # Calculate difference

        elif "<" in con or ">" in con:
            # Case: Less than (<) or greater than (>) values
            con_value = float(re.findall(r"-?\d+\.?\d*", con)[0])

        else:
            # Case: Simple numeric value
            con_value = float(re.findall(r"-?\d+\.?\d*", con)[0])

        return con_value

    except Exception:
        logger.exception(msg="Unexpected error in 'calculate_concentration()'.")
        raise


def transform_concentration(
    new_df: pd.DataFrame,
    logger: CustomLogger,
) -> pd.DataFrame:
    """
    Transform the concentration values in the DataFrame based on the units.

    Parameters
    ----------
    new_df : pd.DataFrame
        A DataFrame containing concentration and unit information.
    logger : CustomLogger
        Logger instance for tracking progress and errors.

    Returns
    -------
    pd.DataFrame
        A DataFrame with updated concentration values.
    """
    try:
        # -------------------- Logging: Start --------------------
        logger.info(
            msg="/ Task: Transform the 'concentration values' units and calculate their logarithmic values."
        )
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")
        logger.log_with_borders(
            level=logging.INFO,
            message="Starting transformation of concentration values based on units.",
            border="|",
            length=100,
        )

        # Identify rows with unit 'μg/ml' and transform their MIC values
        mask = new_df["Unit"] == "μg/ml"
        new_df.loc[mask, "MIC Value"] = (
            pd.to_numeric(new_df.loc[mask, "MIC Value"])  # Ensure MIC Value is numeric
            / new_df.loc[mask, "Molecular Weight"]  # Divide by molecular weight
            * 1000  # Convert to appropriate scale
        )

        # Calculate the logarithmic MIC Value for all rows
        # If MIC Value is zero or negative, use `-inf` as a placeholder
        new_df["Log MIC Value"] = new_df["MIC Value"].apply(
            lambda x: (
                math.log10(x) if x > 0 else float("-inf")
            )  # or use 0 if appropriate
        )

        # -------------------- Logging: Completion --------------------
        logger.log_with_borders(
            level=logging.INFO,
            message="Log MIC Value calculation complete.",
            border="|",
            length=100,
        )
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

        return new_df

    except Exception:
        logger.exception(msg="Unexpected error in 'transform_concentration()'.")
        raise


def filter_save_by_target_strains(
    input_df: pd.DataFrame,
    output_csv: str,
    keyword: str,
    logger: "CustomLogger",
) -> None:
    """
    Filters, saves, and logs information for a specific target keyword from a DataFrame.

    Parameters
    ----------
    input_df : pd.DataFrame
        DataFrame to filter.
    output_csv : str
        File path for saving the filtered DataFrame.
    keyword : str
        Keyword used for filtering the 'Targets' column.
    logger : CustomLogger
        Logger instance for tracking progress and errors.

    Returns
    -------
    None
    """
    # Filter the DataFrame for the current target keyword
    logger.info(
        msg="/ Task: Filter data targeting strain, count occurrences, and save the results."
    )
    logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")
    filtered_df = input_df[input_df["Targets"].str.contains(keyword)]

    # Log the top N target counts with borders
    for target, count in filtered_df.Targets.value_counts()[:10].items():
        logger.log_with_borders(
            level=logging.INFO,
            message=f"{target}: {count}",
            border="|",
            length=100,
        )

    # Add a divider after logging
    logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

    # -------------------- Save output --------------------
    output_dir = os.path.dirname(output_csv)
    os.makedirs(output_dir, exist_ok=True)
    filtered_df.to_csv(path_or_buf=output_csv, index=False)
    logger.log_with_borders(
        level=logging.INFO,
        message=f"Saved:\n'{output_csv}'",
        border="|",
        length=100,
    )
    logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")


def collect_dbaasp(base_path: str, strains: dict, logger: CustomLogger) -> None:
    """
    Processes DBAASP data, filters by strain, and saves the results.

    Parameters
    ----------
    base_path : str
        Base directory for data files.
    strains : dict
        Dictionary mapping full strain names to their suffixes.
    logger : CustomLogger
        Logger instance for tracking progress and errors.

    Returns
    -------
    None
    """
    try:
        # Load DBAASP data
        dbaasp_input_path = os.path.join(
            base_path, "data/raw/peptides-complete1220.csv"
        )
        dbaasp_df = pd.read_csv(filepath_or_buffer=dbaasp_input_path)

        # Select data of monomer type
        dbaasp_df = dbaasp_df[dbaasp_df["COMPLEXITY"] == "Monomer"]

        # Set target activity concentration to 10000 where it is missing
        dbaasp_df.loc[
            dbaasp_df["TARGET ACTIVITY - CONCENTRATION"].isnull(),
            ["TARGET ACTIVITY - CONCENTRATION"],
        ] = "10000"

        # Rename columns
        dbaasp_df = dbaasp_df.rename(
            columns={
                "SEQUENCE": "Sequence",
                "TARGET ACTIVITY - TARGET SPECIES": "Targets",
                "TARGET ACTIVITY - CONCENTRATION": "MIC",
                "TARGET ACTIVITY - UNIT": "Unit",
            }
        )

        # Process the DBAASP sequences
        dbaasp_processed_df = process_sequences(
            df=dbaasp_df,
            lower_bound=5,
            upper_bound=61,
            logger=logger,
        )
        logger.add_spacer()

        # Calculate DBAASP concentration
        concentration = []

        # Iterate through each concentration value in the data
        for con in dbaasp_processed_df["MIC"]:
            if "-" in con:
                # If the concentration contains a '-', extract the first numeric value
                concentration.append(float(re.findall(r"-?\d+\.?\d*", con)[0]))
            elif "±" in con:
                # If the concentration contains '±', calculate the mean value
                a = float(re.findall(r"-?\d+\.?\d*", con)[0])
                b = float(re.findall(r"-?\d+\.?\d*", con)[1])
                c = round(a - b, 3)
                if c < 0:
                    c = a
                concentration.append(c)
            elif "<" in con:
                # If the concentration contains '<', extract the numeric value
                concentration.append(float(re.findall(r"-?\d+\.?\d*", con)[0]))
            elif ">" in con:
                # If the concentration contains '>', extract the numeric value
                concentration.append(float(re.findall(r"-?\d+\.?\d*", con)[0]))
            else:
                # Otherwise, directly extract the numeric value
                concentration.append(float(re.findall(r"-?\d+\.?\d*", con)[0]))

        # Store the new concentration data in the DataFrame
        dbaasp_processed_df["MIC Value"] = concentration

        # Calculate new concentration units (if the original unit is µg/ml)
        new_con = (
            dbaasp_processed_df.loc[dbaasp_processed_df["Unit"] == "µg/ml"]["MIC Value"]
            / dbaasp_processed_df.loc[dbaasp_processed_df["Unit"] == "µg/ml"][
                "Molecular Weight"
            ]
        ) * 1000

        # Update the concentration data in the DataFrame
        dbaasp_processed_df.loc[
            dbaasp_processed_df["Unit"] == "µg/ml", ["MIC Value"]
        ] = new_con

        # Calculate the logarithm of the concentration values
        dbaasp_processed_df["Log MIC Value"] = dbaasp_processed_df["MIC Value"].map(
            lambda x: math.log10(x) if x > 0 else 0
        )

        # Initialize an empty DataFrame to store DBAASP data
        dbaasp = pd.DataFrame()

        # Iterate over each strain
        for _, (strain, _) in enumerate(strains.items(), start=1):
            # Select rows related to the current strain
            new_row = dbaasp_processed_df[
                dbaasp_processed_df["Targets"].str.contains(strain) == True
            ]

            # Append the selected rows to the DBAASP DataFrame
            dbaasp = pd.concat([dbaasp, new_row], ignore_index=True)

        # Add 'DBAASP_' prefix to the IDs
        dbaasp["ID"] = dbaasp["ID"].map(lambda x: f"DBAASP_{x}")

        # Select rows where the activity measure group is 'MIC'
        dbaasp = dbaasp[dbaasp["TARGET ACTIVITY - ACTIVITY MEASURE GROUP"] == "MIC"]

        # Select the required columns
        dbaasp = dbaasp[
            [
                "ID",
                "Sequence",
                "Targets",
                "Sequence Length",
                "Molecular Weight",
                "MIC",
                "MIC Value",
                "Unit",
                "Log MIC Value",
            ]
        ]

        # Filter and save DBAASP data
        for strain_index, (strain, suffix) in enumerate(strains.items(), start=1):
            logger.info(msg=f"/ 1.{strain_index}")
            logger.add_divider(level=logging.INFO, length=40, border="+", fill="-")
            logger.log_with_borders(
                level=logging.INFO,
                message=f"Strain - '{strain}'",
                border="|",
                length=40,
            )
            logger.add_divider(level=logging.INFO, length=40, border="+", fill="-")
            logger.add_spacer(level=logging.INFO, lines=1)
            output_csv = os.path.join(base_path, f"data/interim/{suffix}/DBAASP.csv")
            filter_save_by_target_strains(
                input_df=dbaasp,
                output_csv=output_csv,
                keyword=strain,
                logger=logger,
            )
            logger.add_spacer()

    except Exception:
        logger.exception(msg="Unexpected error in 'collect_dbaasp()'.")
        raise


def collect_dbamp(base_path: str, strains: dict, logger: CustomLogger) -> None:
    """
    Processes dbAMP data, filters by strain, and saves the results.

    Parameters
    ----------
    base_path : str
        Base directory for data files.
    strains : dict
        Dictionary mapping full strain names to their suffixes.
    logger : CustomLogger
        Logger instance for tracking progress and errors.

    Returns
    -------
    None
    """
    try:
        # Load dbAMP data
        dbamp_des_input_path = os.path.join(base_path, "data/raw/dbAMPv2.0.xls")
        dbamp_seq_input_path = os.path.join(base_path, "data/raw/dbamp_raw.csv")
        dbamp_des = pd.read_excel(io=dbamp_des_input_path)
        dbamp_seq = pd.read_csv(filepath_or_buffer=dbamp_seq_input_path)

        # Rename the 'ID' column to 'dbAMP_ID' in the sequence data
        dbamp_seq = dbamp_seq.rename(columns={"ID": "dbAMP_ID"})

        # Merge the descriptive data and sequence data on 'dbAMP_ID'
        dbamp_df = pd.merge(dbamp_seq, dbamp_des, on="dbAMP_ID")

        # Rename the 'sequence' column to 'Sequence'
        dbamp_df = dbamp_df.rename(columns={"sequence": "Sequence"})

        # Remove rows where the 'Targets' column is null
        dbamp_df = dbamp_df[dbamp_df["Targets"].notnull()]

        # Process the dbAMP sequences
        dbamp_processed_df = process_sequences(
            df=dbamp_df,
            lower_bound=5,
            upper_bound=61,
            logger=logger,
        )
        logger.add_spacer()

        # Reset the index of the DataFrame
        dbamp_processed_df = dbamp_processed_df.reset_index(drop=True)

        # Initialize an empty DataFrame to store the results
        dbamp = pd.DataFrame(
            columns=[
                "ID",
                "Sequence",
                "Targets",
                "Sequence Length",
                "Molecular Weight",
                "MIC",
                "MIC Value",
                "Unit",
                "Log MIC Value",
            ]
        )

        # Define a regular expression to match MIC values
        regex = r"\(MIC=\S+\)"

        # Iterate over each strain
        for _, (strain, _) in enumerate(strains.items(), start=1):
            for i in range(len(dbamp_processed_df)):
                # Extract information related to the strain
                res1 = [
                    string
                    for string in dbamp_processed_df["Targets"][i].split("&&")
                    if strain in string
                ]
                # If multiple strain-related MIC values are found, prioritize the first one
                if len(res1) >= 2:
                    final = []
                    # Iterate over each matched result
                    for _, item in enumerate(res1):
                        if "MIC" not in item:
                            continue
                        # Find the MIC value using the regular expression and calculate the concentration
                        res2 = re.search(regex, item)
                        if res2 is None:
                            continue
                        res3 = calculate_concentration(con=res2.group(0), logger=logger)
                        x = (res2.group(0), res3)
                        final.append(x)
                    # If no valid MIC value is found, continue to the next iteration
                    if not final:
                        continue
                    # sorted_by_second = sorted(final, key=lambda tup: tup[1])
                    res2 = final[0]
                    res3 = final[0][1]
                else:
                    if len(res1) == 0:
                        continue

                    if "MIC" not in res1[0]:
                        continue

                    res2 = re.search(regex, res1[0])
                    if res2 is None:
                        continue

                    res3 = calculate_concentration(con=res2.group(0), logger=logger)

                # Determine the unit of concentration
                unit = "µM" if "µM" in res2[0] else "μg/ml"
                # Construct a new row of data
                new_row = {
                    "ID": dbamp_processed_df.loc[i].dbAMP_ID,
                    "Sequence": dbamp_processed_df.loc[i].Sequence,
                    "Targets": strain,
                    "Sequence Length": dbamp_processed_df.loc[i]["Sequence Length"],
                    "Molecular Weight": dbamp_processed_df.loc[i]["Molecular Weight"],
                    "MIC": res2[0],
                    "MIC Value": res3,
                    "Unit": unit,
                }

                # Add the new row to the DataFrame
                dbamp = pd.concat([dbamp, pd.DataFrame([new_row])], ignore_index=True)

        # Transform the concentration values' units and calculate their logarithmic values
        dbamp = transform_concentration(new_df=dbamp, logger=logger)
        logger.add_spacer()

        # Filter and save dbAMP data
        for strain_index, (strain, suffix) in enumerate(strains.items(), start=1):
            logger.info(msg=f"/ 2.{strain_index}")
            logger.add_divider(level=logging.INFO, length=40, border="+", fill="-")
            logger.log_with_borders(
                level=logging.INFO,
                message=f"Strain - '{strain}'",
                border="|",
                length=40,
            )
            logger.add_divider(level=logging.INFO, length=40, border="+", fill="-")
            logger.add_spacer(level=logging.INFO, lines=1)
            output_csv = os.path.join(base_path, f"data/interim/{suffix}/dbAMP.csv")
            filter_save_by_target_strains(
                input_df=dbamp,
                output_csv=output_csv,
                keyword=strain,
                logger=logger,
            )
            logger.add_spacer()

    except Exception:
        logger.exception(msg="Unexpected error in 'collect_dbamp()'.")
        raise


def collect_dramp(base_path: str, strains: dict, logger: CustomLogger) -> None:
    """
    Processes dbAMP data, filters by strain, and saves the results.

    Parameters
    ----------
    base_path : str
        Base directory for data files.
    strains : dict
        Dictionary mapping full strain names to their suffixes.
    logger : CustomLogger
        Logger instance for tracking progress and errors.

    Returns
    -------
    None
    """
    try:
        # Load Dramp data
        dramp_df_input_path = os.path.join(base_path, "data/raw/DRAMP.xlsx")
        dramp_df = pd.read_excel(io=dramp_df_input_path)

        # Rename the 'Sequence' column to 'Sequence'
        dramp_df = dramp_df.rename(columns={"Sequence": "Sequence"})

        # Process the Dramp sequences
        dramp_processed_df = process_sequences(
            df=dramp_df,
            lower_bound=5,
            upper_bound=61,
            logger=logger,
        )
        logger.add_spacer()

        # Reset the index
        dramp_processed_df = dramp_processed_df.reset_index(drop=True)

        # Insert an underscore '_' between "DRAMP" and the numbers in each value of the 'DRAMP_ID' column
        dramp_processed_df["DRAMP_ID"] = dramp_processed_df["DRAMP_ID"].str.replace(
            "DRAMP", "DRAMP_", regex=True
        )

        # Initialize an empty DataFrame to store the results
        dramp = pd.DataFrame(
            columns=[
                "ID",
                "Sequence",
                "Targets",
                "Sequence Length",
                "Molecular Weight",
                "MIC",
                "MIC Value",
                "Unit",
                "Log MIC Value",
            ]
        )

        # Define a regular expression to match MIC values
        regex = r"\(MIC=\S+\)"

        # Iterate over each strain
        for _, (strain, _) in enumerate(strains.items(), start=1):
            for _, row in dramp_processed_df.iterrows():
                # Extract information related to the strain
                res1 = [
                    string
                    for string in re.split(",| ;|##", row["Target_Organism"])
                    if strain in string
                ]
                # If no strain-related information is found, continue to the next iteration
                if len(res1) == 0:
                    continue
                # If strain-related information is found
                if len(res1) >= 2:
                    final = []
                    for _, item in enumerate(res1):
                        if "MIC" not in item:
                            continue
                        # Find the MIC value using the regular expression and calculate the concentration
                        res2 = re.search(regex, item)
                        if res2 is None:
                            continue
                        res3 = calculate_concentration(con=res2.group(0), logger=logger)
                        x = (res2.group(0), res3)
                        final.append(x)
                    if not final:
                        continue
                    # sorted_by_second = sorted(final, key=lambda tup: tup[1])
                    res2 = final[0]
                    res3 = final[0][1]
                else:
                    if len(res1) == 0:
                        continue

                    if "MIC" not in res1[0]:
                        continue

                    res2 = re.search(regex, res1[0])
                    if res2 is None:
                        continue

                    res3 = calculate_concentration(con=res2.group(0), logger=logger)

                # Determine the unit of concentration
                unit = (
                    "μM"
                    if "μM" in res2[0]
                    else (
                        "μg/ml"
                        if ("μg/ml" in res2[0]) or ("μg/mL" in res2[0])
                        else "NONE"
                    )
                )
                # Construct a new row of data
                new_row = {
                    "ID": row.DRAMP_ID,
                    "Sequence": row.Sequence,
                    "Targets": strain,
                    "Sequence Length": row["Sequence Length"],
                    "Molecular Weight": row["Molecular Weight"],
                    "MIC": res2[0],
                    "MIC Value": res3,
                    "Unit": unit,
                }
                # Add the new row to the DataFrame
                dramp = pd.concat([dramp, pd.DataFrame([new_row])], ignore_index=True)

        # Remove rows where the concentration unit could not be determined
        dramp = dramp[dramp["Unit"] != "NONE"]

        # Transform the concentration values' units and calculate their logarithmic values
        dramp = transform_concentration(new_df=dramp, logger=logger)
        logger.add_spacer()

        # Filter and save Dramp data
        for strain_index, (strain, suffix) in enumerate(strains.items(), start=1):
            logger.info(msg=f"/ 3.{strain_index}")
            logger.add_divider(level=logging.INFO, length=40, border="+", fill="-")
            logger.log_with_borders(
                level=logging.INFO,
                message=f"Strain - '{strain}'",
                border="|",
                length=40,
            )
            logger.add_divider(level=logging.INFO, length=40, border="+", fill="-")
            logger.add_spacer(level=logging.INFO, lines=1)
            output_csv = os.path.join(base_path, f"data/interim/{suffix}/Dramp.csv")
            filter_save_by_target_strains(
                input_df=dramp,
                output_csv=output_csv,
                keyword=strain,
                logger=logger,
            )
            logger.add_spacer()

    except Exception:
        logger.exception(msg="Unexpected error in 'collect_dramp()'.")
        raise


# ============================== Pipeline Entry Point ==============================
def run_collect_pipeline(base_path: str, logger: CustomLogger) -> None:
    """
    Entry point for the data collection pipeline. This function coordinates the
    extraction and preprocessing of antimicrobial peptide (AMP) data from multiple
    public databases, including DBAASP, dbAMP, and DRAMP. The processed datasets
    are filtered by target strains and saved for downstream analysis.

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

        # ---------- Collect DBAASP ----------
        logger.info(msg="/ 1")
        logger.add_divider(level=logging.INFO, length=30, border="+", fill="-")
        logger.log_with_borders(
            level=logging.INFO,
            message="Database - 'DBAASP'",
            border="|",
            length=30,
        )
        logger.add_divider(level=logging.INFO, length=30, border="+", fill="-")
        logger.add_spacer(level=logging.INFO, lines=1)
        collect_dbaasp(base_path=base_path, strains=strains, logger=logger)

        # ---------- Collect dbAMP ----------
        logger.info(msg="/ 2")
        logger.add_divider(level=logging.INFO, length=30, border="+", fill="-")
        logger.log_with_borders(
            level=logging.INFO,
            message="Database - 'dbAMP'",
            border="|",
            length=30,
        )
        logger.add_divider(level=logging.INFO, length=30, border="+", fill="-")
        logger.add_spacer(level=logging.INFO, lines=1)
        collect_dbamp(base_path=base_path, strains=strains, logger=logger)

        # ---------- Collect Dramp ----------
        logger.info(msg="/ 3")
        logger.add_divider(level=logging.INFO, length=30, border="+", fill="-")
        logger.log_with_borders(
            level=logging.INFO,
            message="Database - 'Dramp'",
            border="|",
            length=30,
        )
        logger.add_divider(level=logging.INFO, length=30, border="+", fill="-")
        logger.add_spacer(level=logging.INFO, lines=1)
        collect_dramp(base_path=base_path, strains=strains, logger=logger)

        # ---------- Merge DBAASP + dbAMP + Dramp ----------
        logger.info(msg="/ 4")
        logger.add_divider(level=logging.INFO, length=50, border="+", fill="-")
        logger.log_with_borders(
            level=logging.INFO,
            message="Database - 'DBAASP + dbAMP + Dramp'",
            border="|",
            length=50,
        )
        logger.add_divider(level=logging.INFO, length=50, border="+", fill="-")
        logger.add_spacer(level=logging.INFO, lines=1)

        # Merge and save data for each strain
        for _, (strain, suffix) in enumerate(strains.items(), start=1):

            # Log the current task with the strain being processed
            logger.info(msg=f"/ Task: Merge and save all data targeting '{strain}'.")

            # Load data for the current strain from respective sources
            df_dbaasp_input_file = os.path.join(
                base_path, f"data/interim/{suffix}/DBAASP.csv"
            )
            df_dbamp_input_file = os.path.join(
                base_path, f"data/interim/{suffix}/dbAMP.csv"
            )
            df_dramp_input_file = os.path.join(
                base_path, f"data/interim/{suffix}/Dramp.csv"
            )
            df_dbaasp = pd.read_csv(filepath_or_buffer=df_dbaasp_input_file)
            df_dbamp = pd.read_csv(filepath_or_buffer=df_dbamp_input_file)
            df_dramp = pd.read_csv(filepath_or_buffer=df_dramp_input_file)

            # Log the start of dataset concatenation
            logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")
            logger.log_with_borders(
                level=logging.INFO,
                message=f"Concatenating datasets for '{strain}'.",
                border="|",
                length=100,
            )

            # Concatenate all datasets into a single DataFrame
            df = pd.concat([df_dbaasp, df_dbamp, df_dramp], axis=0, ignore_index=True)

            # Log dataset shapes for verification
            logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")
            logger.log_with_borders(
                level=logging.INFO,
                message=f"DBAASP: {df_dbaasp.shape[0]}, dbAMP: {df_dbamp.shape[0]}, Dramp: {df_dramp.shape[0]}",
                border="|",
                length=100,
            )
            logger.log_with_borders(
                level=logging.INFO,
                message=f"Concatenated dataset: {df.shape}.",
                border="|",
                length=100,
            )

            # Save the concatenated DataFrame to a CSV file
            df_output_file = os.path.join(
                base_path, f"data/interim/{suffix}/All_Integrated.csv"
            )
            df.to_csv(
                path_or_buf=df_output_file,
                index=False,
            )

            # Log the successful save of the concatenated dataset
            logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")
            logger.log_with_borders(
                level=logging.INFO,
                message=f"Saved:\n'{df_output_file}'",
                border="|",
                length=100,
            )

            # Add a divider and a newline for better log readability
            logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")
            logger.add_spacer(level=logging.INFO, lines=1)

    except Exception:
        logger.exception(msg="Unexpected error in 'run_collect_pipeline()'.")
        raise
