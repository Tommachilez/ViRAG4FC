#!.venv/Scripts/python
# document_retrieval_refactored.py
# -*- coding: utf-8 -*-

"""
Main script to orchestrate document retrieval.

This script can be run in two modes:
1.  Single Query Mode (Default): Processes a single search query provided
    via the command line.
    Example:
    $ python document_retrieval_refactored.py --query "Benefits of Python" \
      --search-output-base "output/single/raw_results.jsonl" \
      --extracted-output-base "output/single/extracted.jsonl"

2.  Batch File Mode: Reads queries from a specified input file, processes
    them in batches, and saves the results into versioned batch files.
    Example:
    $ python document_retrieval_refactored.py --input-file "queries.csv" \
      --query-column "claim" --search-output-base "output/batch/raw" \
      --extracted-output-base "output/batch/text"
"""

import argparse
import logging
import time
import math
from pathlib import Path
from typing import Dict, List, Generator, Any

import requests
from tqdm import tqdm

import config
from google_cse import search_google_cse, process_search_results
from content_extractor import extract_main_text_requests
from data_handler import read_data_from_file, save_jsonl_record


# --- Main Application Logic ---

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(module)s] - %(message)s'
)


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Search Google CSE, save raw results, extract text, and save final results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("-q", "--query", type=str, help="A single search query to process.")
    input_group.add_argument("-i", "--input-file", type=str, help="Input CSV/TSV file containing queries.")

    parser.add_argument("--query-column", type=str, help="Column name for search queries (required for --input-file).")
    parser.add_argument("--num-queries", type=int, default=None, help="Number of queries from the input file to process.")

    parser.add_argument("-p", "--pages", type=int, default=1, help="Number of Google Search result pages per query.")
    parser.add_argument("-n", "--num-results", type=int, default=config.DEFAULT_CSE_NUM_RESULTS, help="Results per CSE page (1-10).")
    parser.add_argument("-d", "--delay", type=float, default=config.DEFAULT_REQUEST_DELAY, help="Delay between CSE API requests (secs).")
    parser.add_argument("-t", "--timeout", type=int, default=config.DEFAULT_EXTRACTION_TIMEOUT, help="Timeout for URL text extraction (secs).")
    parser.add_argument("--site-search", type=str, default=None, help="Restrict search to a specific site.")
    parser.add_argument("--file-type", type=str, default=None, help="Restrict search to a specific file type (e.g., pdf, docx).")
    parser.add_argument("--batch-size", type=int, default=10, help="Queries per batch when using --input-file.")

    parser.add_argument("--search-output-base", type=str, default=config.DEFAULT_SEARCH_OUTPUT_BASE, help="Base path for raw search files. Batch mode: a prefix. Single mode: a timestamp is added for uniqueness.")
    parser.add_argument("--extracted-output-base", type=str, default=config.DEFAULT_EXTRACTED_OUTPUT_BASE, help="Base path for extracted text files. Batch mode: a prefix. Single mode: a timestamp is added for uniqueness.")

    args = parser.parse_args()

    if args.input_file and not args.query_column:
        parser.error("--query-column is required when using --input-file.")
    if args.num_queries is not None and not args.input_file:
        parser.error("--num-queries can only be used with --input-file.")
    if not 1 <= args.num_results <= 10:
        logging.warning("Num results (%d) out of range (1-10). Clamping to 10.", args.num_results); args.num_results = 10
    if args.batch_size <= 0:
        logging.warning("Batch size (%d) must be positive. Setting to 1.", args.batch_size); args.batch_size = 1
    if args.num_queries is not None and args.num_queries <= 0:
        logging.warning("Number of queries (%d) must be positive. Ignoring.", args.num_queries)
        args.num_queries = None

    return args


def batch_generator(data: List[Any], batch_size: int) -> Generator[List[Any], None, None]:
    for i in range(0, len(data), batch_size): yield data[i:i + batch_size]


def _process_single_query(
    query: str, args: argparse.Namespace
) -> tuple[Dict[str, int], List[Dict], List[Dict]]:
    counters = {"raw_saved": 0, "urls_processed": 0, "extractions_success": 0}
    raw_results_for_query, extracted_results_for_query = [], []
    stop_fetching_pages = False
    query_short = (query[:35] + '...') if len(query) > 35 else query

    for page in range(args.pages):
        if stop_fetching_pages: break
        start_index = page * args.num_results + 1
        logging.debug("Fetching page %d for query '%s'", page + 1, query_short)

        try:
            api_kwargs = {'siteSearch': args.site_search} if args.site_search else {}
            results_json = search_google_cse(
                query=query,
                api_key=config.API_KEY,
                cse_id=config.CSE_ID,
                num_results=args.num_results,
                start_index=start_index,
                include_file_type=args.file_type,
                **api_kwargs
            )
        except requests.exceptions.RequestException as e:
            logging.error("Stopping for query '%s' due to CSE API network error: %s", query_short, e); break

        if results_json is None:
            logging.warning("No results or error from CSE API for query '%s', page %d.", query_short, page + 1); break

        search_results_list = process_search_results(results_json)
        if not search_results_list:
            logging.info("No valid URLs found for query '%s', page %d.", query_short, page + 1); break

        logging.debug("Processing %d results from page %d for query '%s'", len(search_results_list), page + 1, query_short)

        for result_index, search_item in enumerate(search_results_list):
            current_rank = start_index + result_index
            target_url = search_item.get('link')

            raw_record = search_item.copy()
            raw_record.update({'query': query, 'search_page': page + 1, 'approx_rank': current_rank})
            raw_results_for_query.append(raw_record)
            counters["raw_saved"] += 1

            if target_url:
                counters["urls_processed"] += 1
                extracted_content = extract_main_text_requests(target_url, timeout=args.timeout)
                if extracted_content:
                    counters["extractions_success"] += 1
                else:
                    logging.debug("Extraction failed for URL: %s", target_url)
            else:
                extracted_content = None
                logging.debug("Skipping extraction for item rank %d (no link).", current_rank)

            extracted_record = {"query": query, "search_page": page + 1, "approx_rank": current_rank, "url": target_url, "title": search_item.get('title', 'N/A'), "extracted_text": extracted_content}
            extracted_results_for_query.append(extracted_record)

        if len(search_results_list) < args.num_results:
            logging.info("Fewer results than requested; stopping page fetch for '%s'.", query_short)
            stop_fetching_pages = True

    logging.info("Query '%s' finished. Results: %d. URLs processed: %d (%d successful).", query_short, counters["raw_saved"], counters["urls_processed"], counters["extractions_success"])
    return counters, raw_results_for_query, extracted_results_for_query


def _save_results_to_file(filepath: Path, results: List[Dict]) -> int:
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        lines_written = 0
        with open(filepath, 'w', encoding='utf-8') as f:
            for record in results:
                if save_jsonl_record(f, record): lines_written += 1
        logging.info(f"Successfully wrote {lines_written}/{len(results)} records to {filepath}")
        return lines_written
    except IOError as e:
        logging.error(f"Failed to open or write to file {filepath}: {e}"); return 0


def _get_unique_filepath(base_path_str: str) -> Path:
    """Appends a timestamp to a filename to make it unique."""
    base_path = Path(base_path_str)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    directory = base_path.parent
    stem = base_path.stem
    suffix = base_path.suffix or '.jsonl'  # Default to .jsonl if no extension
    
    unique_filename = f"{timestamp}_{stem}{suffix}"
    return directory / unique_filename


def process_single_cli_query(args: argparse.Namespace):
    """Handles the entire process for a single query from the command line."""
    logging.info(f"Starting single query processing for: '{args.query}'")
    
    counters, raw_results, extracted_results = _process_single_query(args.query, args)
    
    # --- Generate unique file paths and save results ---
    raw_output_path = _get_unique_filepath(args.search_output_base)
    extracted_output_path = _get_unique_filepath(args.extracted_output_base)
    
    raw_written = _save_results_to_file(raw_output_path, raw_results)
    extracted_written = _save_results_to_file(extracted_output_path, extracted_results)
    
    # --- Final Summary ---
    print("\n--- Single Query Processing Summary ---")
    print(f"Query: '{args.query}'")
    print(f"Total raw search results generated: {counters['raw_saved']} (wrote {raw_written})")
    print(f"Total URLs processed for extraction: {counters['urls_processed']}")
    print(f"Total successful text extractions: {counters['extractions_success']}")
    print(f"Raw search results saved to: {raw_output_path}")
    print(f"Extracted text saved to: {extracted_output_path}")
    print("-------------------------------------\n")


def process_file_in_batches(args: argparse.Namespace):
    """Coordinates fetching, extraction, and saving for queries from a file."""
    queries = read_data_from_file(
        filepath=args.input_file, 
        column=args.query_column, 
        num_rows=args.num_queries
    )

    if not queries:
        logging.error("No queries to process from file. Exiting.")
        return

    try:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        raw_base_path = Path(args.search_output_base)
        raw_batch_dir = raw_base_path.parent / f"{timestamp}_{raw_base_path.name}_batches"
        raw_batch_dir.mkdir(parents=True, exist_ok=True)
        
        extracted_base_path = Path(args.extracted_output_base)
        extracted_batch_dir = extracted_base_path.parent / f"{timestamp}_{extracted_base_path.name}_batches"
        extracted_batch_dir.mkdir(parents=True, exist_ok=True)
        
        logging.info(f"Raw search batch results will be saved in: {raw_batch_dir}/")
        logging.info(f"Extracted text batch results will be saved in: {extracted_batch_dir}/")
    except OSError as e:
        logging.error(f"Failed to create output directories: {e}. Exiting."); return

    total_counters = {"raw_saved": 0, "urls_processed": 0, "extractions_success": 0}
    total_lines_written = {"raw": 0, "extracted": 0}
    
    num_queries = len(queries)
    num_batches = math.ceil(num_queries / args.batch_size)
    query_batches = batch_generator(queries, args.batch_size)
    
    logging.info("Processing %d queries in %d batches of size up to %d.", num_queries, num_batches, args.batch_size)

    batch_iterator = tqdm(query_batches, total=num_batches, desc="Processing Batches", unit="batch")
    for batch_index, query_batch in enumerate(batch_iterator):
        batch_raw_results, batch_extracted_results = [], []

        for query_index, query in enumerate(query_batch):
            if (batch_index > 0 or query_index > 0) and args.delay > 0:
                time.sleep(args.delay)
            
            q_counters, q_raw, q_extracted = _process_single_query(query, args)
            
            for key in total_counters: total_counters[key] += q_counters.get(key, 0)
            batch_raw_results.extend(q_raw)
            batch_extracted_results.extend(q_extracted)
        
        logging.info(f"Batch {batch_index + 1}/{num_batches} complete. Saving results...")
        
        raw_batch_path = raw_batch_dir / f"{raw_base_path.name}_{batch_index + 1}.jsonl"
        extracted_batch_path = extracted_batch_dir / f"{extracted_base_path.name}_{batch_index + 1}.jsonl"
        
        total_lines_written["raw"] += _save_results_to_file(raw_batch_path, batch_raw_results)
        total_lines_written["extracted"] += _save_results_to_file(extracted_batch_path, batch_extracted_results)
        
    print("\n--- Batch File Processing Summary ---")
    print(f"Input Source: '{args.input_file}' (Column: '{args.query_column}')")
    print(f"Processed {num_queries} total queries in {num_batches} batches.")
    print(f"Total raw lines written to files: {total_lines_written['raw']}")
    print(f"Total extracted lines written to files: {total_lines_written['extracted']}")
    print(f"Raw search batch files saved in: {raw_batch_dir}/")
    print(f"Extracted text batch files saved in: {extracted_batch_dir}/")
    print("-----------------------------------\n")


def main():
    """Main entry point for the script."""
    args = parse_arguments()
    try:
        config.validate_config()
        logging.info("API Key and CSE ID loaded successfully.")
    except ValueError as e:
        logging.error("%s. Exiting.", e); return

    try:
        if args.query:
            process_single_cli_query(args)
        elif args.input_file:
            process_file_in_batches(args)
    except Exception as e:
        logging.error("An unexpected fatal error occurred during main execution: %s", e, exc_info=True)
    finally:
        logging.info("Script finished.")


if __name__ == "__main__":
    main()