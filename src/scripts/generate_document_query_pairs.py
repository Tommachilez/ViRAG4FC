#!.venv/Scripts/python
import csv
import os
from typing import List


def process_csvs_to_tsv(csv_file_paths: List[str], output_tsv_path: str):
    """
    Reads a list of CSV files, extracts document/query pairs, and saves
    them to a single Tab-Separated Values (TSV) file.

    This function automatically detects whether the query column is named
    'statement' or 'claim'.

    Args:
        csv_file_paths: A list of strings, where each string is a path to a CSV file.
        output_tsv_path: The path for the output TSV file.
    """
    print("Starting data processing...")
    extracted_pairs = []

    for file_path in csv_file_paths:
        # First, check if the file actually exists to avoid errors
        if not os.path.exists(file_path):
            print(f"⚠️ Warning: File not found, skipping: {file_path}")
            continue

        try:
            with open(file_path, mode='r', encoding='utf-8') as infile:
                # Use DictReader to easily access columns by their header names
                reader = csv.DictReader(infile)
                headers = reader.fieldnames

                # --- Smart Column Detection ---
                # Determine the name of the 'query' column ('statement' or 'claim')
                query_column = None
                if 'statement' in headers:
                    query_column = 'statement'
                elif 'claim' in headers:
                    query_column = 'claim'

                # Ensure the required 'context' and a query column exist
                if 'context' not in headers or not query_column:
                    print(f"⚠️ Warning: Skipping '{file_path}' because it lacks the required 'context' and/or 'statement'/'claim' columns.")
                    continue

                print(f"Processing '{file_path}'... (Document: 'context', Query: '{query_column}')")

                # --- Data Extraction ---
                for row in reader:
                    document = row.get('context', '').strip()
                    query = row.get(query_column, '').strip()
                    
                    # Clean any internal tabs from the data by replacing them with a space
                    document = document.replace('\t', ' ').replace('\n', ' ')
                    query = query.replace('\t', ' ').replace('\n', ' ')

                    # Add the pair only if both document and query have content
                    if document and query:
                        extracted_pairs.append([document, query])

        except Exception as e:
            print(f"❌ Error processing file {file_path}: {e}")

    # --- Writing to TSV File ---
    if not extracted_pairs:
        print("\nNo data was extracted. The output file will not be created.")
        return

    try:
        print(f"\nWriting {len(extracted_pairs)} pairs to '{output_tsv_path}'...")
        with open(output_tsv_path, mode='w', encoding='utf-8', newline='') as outfile:
            # Use the csv writer with a tab delimiter for TSV format
            writer = csv.writer(outfile, delimiter='\t')
            writer.writerows(extracted_pairs)
        print("✅ Success! Data processing is complete.")
    except Exception as e:
        print(f"❌ Error writing to output file {output_tsv_path}: {e}")


def main():
    csv_files_to_process = [
        "data/vifactcheck/vifactcheck_relevant_pairs.csv",
        "data/viwikifc/viwikifc_relevant_pairs.csv",
        "data/vinumfcr/vinumfcr_relevant_pairs.csv"
    ]

    # 2. DEFINE THE NAME OF THE OUTPUT FILE
    output_filename = "data/document_query_pairs.tsv"

    # 3. RUN THE PROCESSING FUNCTION
    process_csvs_to_tsv(csv_files_to_process, output_filename)


if __name__ == "__main__":
    main()
