# data_handler.py
# -*- coding: utf-8 -*-

"""
Functions for handling data input (reading queries) and potentially output (JSONL writing).
"""

import os
import logging
import json
from typing import List, Optional, Dict, TextIO

import pandas as pd

# def read_queries_from_vifactcheck(column_name: str) -> Optional[List[str]]:
#     """
#     Reads a specified column from a CSV or Excel file into a list of unique queries.

#     Args:
#         file_path: Path to the input dataset file (CSV or Excel).
#         column_name: Name of the column containing search queries.

#     Returns:
#         A list of unique, non-empty queries as strings, or None if an error occurs.
#     """
#     logging.info("Reading queries from column '%s'.", column_name)
#     try:
#         ds = load_dataset("tranthaihoa/vifactcheck")
#         combined = concatenate_datasets([ds['train'], ds['dev'], ds['test']])
#         df = combined.to_pandas()

#         if column_name not in df.columns:
#             logging.error("Query column '%s' not found. Available columns: %s",
#                           column_name, df.columns.tolist())
#             return None # Indicate column not found error

#         # Get unique, non-null queries, convert to string just in case
#         queries = df[column_name].dropna().astype(str).unique().tolist()

#         # Filter out potentially empty strings after conversion
#         queries = [q for q in queries if q.strip()]

#         logging.info("Read %d unique, non-empty queries from column '%s'.", len(queries), column_name)
#         if not queries:
#             logging.warning("No valid, non-empty queries found in the specified column.")
#             # Return empty list instead of None if no queries found but file/column were ok
#             return []
#         return queries

#     except Exception as e:
#         logging.error("Error reading input: %s", e, exc_info=True)
#         return None # Indicate other reading error


def read_data_from_file(filepath: str, column: str, num_rows: Optional[int] = None) -> Optional[List[str]]:
    """
    Reads a specific column of data from CSV, TSV, JSON, or JSONL files.

    This function leverages the powerful pandas library to robustly handle different
    file formats and common data reading errors.

    Args:
        filepath (str): The path to the input file. The function infers the
                        file type from the extension (.csv, .tsv, .json, .jsonl).
        column (str): The name of the column to extract.
        num_rows (Optional[int]): The number of rows to read from the file.
                                  If None (the default), all rows will be read.

    Returns:
        Optional[List[str]]: A list of strings containing the data from the
                             specified column, or None if an error occurred.
    """
    try:
        # Check if the file exists
        if not os.path.exists(filepath):
            logging.error(f"File not found: {filepath}")
            return None

        # Validate num_rows if provided
        if num_rows is not None and num_rows <= 0:
            logging.warning("num_rows must be a positive integer. Reading all rows instead.")
            num_rows = None

        # Determine the file type based on the extension and read it
        file_extension = os.path.splitext(filepath)[1].lower()
        df = None # Initialize DataFrame

        if file_extension in ['.csv', '.tsv']:
            sep = ',' if file_extension == '.csv' else '\t'
            file_type = file_extension[1:].upper()
            logging.info(f"Reading file as {file_type}... (rows: {'all' if num_rows is None else num_rows})")
            # Use the efficient 'nrows' parameter for CSV/TSV files
            df = pd.read_csv(filepath, sep=sep, nrows=num_rows)

        elif file_extension in ['.json', '.jsonl']:
            lines = file_extension == '.jsonl'
            file_type = "JSON Lines" if lines else "JSON"
            logging.info(f"Reading file as {file_type}...")
            # read_json does not have a universal 'nrows', so we read and then slice
            temp_df = pd.read_json(filepath, lines=lines)
            if num_rows is not None:
                logging.info(f"Selecting the first {min(num_rows, len(temp_df))} rows.")
                df = temp_df.head(num_rows)
            else:
                df = temp_df
        else:
            logging.error(f"Unsupported file format: {file_extension}")
            return None

        # Check if the specified column exists in the DataFrame
        if column not in df.columns:
            logging.error(f"Column '{column}' not found in file: {filepath}")
            logging.info(f"Available columns are: {list(df.columns)}")
            return None

        # Extract the column data and convert it to a list of strings
        extracted_data = df[column].astype(str).tolist()
        logging.info(f"Successfully extracted {len(extracted_data)} items from the '{column}' column.")
        return extracted_data

    except pd.errors.EmptyDataError:
        logging.error(f"The file {filepath} is empty.")
        return [] # Return an empty list for empty files
    except pd.errors.ParserError:
        logging.error(f"The file {filepath} could not be parsed. It may be malformed.")
    except Exception as e:
        logging.error(f"An unexpected error occurred while reading {filepath}: {e}")

    return None


def save_jsonl_record(outfile: TextIO, record: Dict) -> bool:
    """
    Safely serializes a dictionary and writes it as a line to an open JSONL file handle.

    Args:
        outfile: An open text file handle in write/append mode with UTF-8 encoding.
        record: The dictionary to write.

    Returns:
        True if writing was successful, False otherwise.
    """
    try:
        # ensure_ascii=False is important for non-English characters
        json_line = json.dumps(record, ensure_ascii=False)
        outfile.write(json_line + '\n')
        return True
    except Exception as e:
        # Log error but don't stop the entire process for one failed write
        logging.error("Failed to serialize/write record for URL '%s': %s", record.get('url', 'N/A'), e)
        return False
