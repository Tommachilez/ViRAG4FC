# main.py
# -*- coding: utf-8 -*-

"""
Main script to orchestrate the process:
1. Read queries from a dataset.
2. Process queries in batches: For each query, perform Google Search & extract text.
3. After each batch completes, save its raw results and extracted results
   into separate files within automatically created directories.
"""

import argparse
import logging
import time
import math
import os
from typing import Dict, List, Generator, Any

import requests
from tqdm import tqdm

import config
from google_cse import search_google_cse, process_search_results
from text_extractor import extract_main_text_requests
from data_handler import read_queries_from_vifactcheck, save_jsonl_record


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(module)s] - %(message)s'
)


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Search Google CSE, save raw results, extract text, save final results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Input
    # parser.add_argument("--input-file", required=True, type=str, help="Input CSV/TSV file containing queries.")
    parser.add_argument("--query-column", required=True, type=str, help="Column name containing search queries.")
    # Search & Extraction Params
    parser.add_argument("-p", "--pages", type=int, default=1, help="N of Google Search result pages per query.")
    parser.add_argument("-n", "--num-results", type=int, default=config.DEFAULT_CSE_NUM_RESULTS, help="Results per CSE page (1-10).")
    parser.add_argument("-d", "--delay", type=float, default=config.DEFAULT_REQUEST_DELAY, help="Delay between CSE API requests (secs).")
    parser.add_argument("-t", "--timeout", type=int, default=config.DEFAULT_EXTRACTION_TIMEOUT, help="Timeout for URL text extraction (secs).")
    parser.add_argument("--site-search", type=str, default=None, help="Restrict search to a specific site.")
    parser.add_argument("--batch-size", type=int, default=10, help="N of queries to process between progress updates.")
    # Output Base Paths (used to determine directory and file naming)
    parser.add_argument("--search-output-base", type=str, default=config.DEFAULT_SEARCH_OUTPUT_BASE, help="Base path and prefix for raw search batch files (e.g., 'out/raw' -> out/raw_batches/raw_1.jsonl).")
    parser.add_argument("--extracted-output-base", type=str, default=config.DEFAULT_EXTRACTED_OUTPUT_BASE, help="Base path and prefix for extracted text batch files (e.g., 'out/text' -> out/text_batches/text_1.jsonl).")

    args = parser.parse_args()
    # Validate num_results
    if not 1 <= args.num_results <= 10:
        logging.warning("Num results (%d) out of range (1-10). Setting to 10.", args.num_results)
        args.num_results = 10
    # Validate batch_size
    if args.batch_size <= 0:
        logging.warning("Batch size (%d) must be positive. Setting to 1.", args.batch_size)
        args.batch_size = 1
    return args


# Helper function to create batches
def batch_generator(data: List[Any], batch_size: int) -> Generator[List[Any], None, None]:
    """Yields successive batches from a list."""
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]


def _process_single_query(
    query: str,
    args: argparse.Namespace
) -> Dict[str, int]:
    """
    Processes all pages for a single search query. Fetches results, extracts text,
    and returns counters and lists of results (does NOT write to files).

    Args:
        query: The search query string.
        args: Parsed command-line arguments.

    Returns:
        A tuple containing:
        - counters: {"raw_saved": int, "urls_processed": int, "extractions_success": int}
        - raw_results: List of raw search result dictionaries for this query.
        - extracted_results: List of extracted text result dictionaries for this query.
    """
    counters = {"raw_saved": 0, "urls_processed": 0, "extractions_success": 0}
    raw_results_for_query: List[Dict] = []
    extracted_results_for_query: List[Dict] = []
    stop_fetching_pages = False
    query_short = (query[:35] + '...') if len(query) > 35 else query

    # --- Loop through Search Pages ---
    for page in range(args.pages):
        if stop_fetching_pages:
            logging.debug("Stopping page fetching for query '%s' as requested.", query_short)
            break

        start_index = page * args.num_results + 1
        logging.debug("Fetching page %d (start index %d) for query '%s'", page + 1, start_index, query_short)

        # --- Fetch Search Results ---
        api_kwargs = {'siteSearch': args.site_search} if args.site_search else {}
        try:
            results_json = search_google_cse(
                query=query, api_key=config.API_KEY, cse_id=config.CSE_ID,
                exclude_file_type="pdf", num_results=args.num_results,
                start_index=start_index, **api_kwargs
            )
        except requests.exceptions.RequestException as e:
            logging.error("Stopping processing for query '%s' due to CSE API network error: %s", query_short, e)
            stop_fetching_pages = True
            continue

        if results_json is None:
            logging.warning("No results or error from CSE API for query '%s', page %d. Stopping page fetch.", query_short, page + 1)
            stop_fetching_pages = True
            continue

        # --- Process & Save Raw Search Results ---
        search_results_list = process_search_results(results_json)
        if search_results_list:
            logging.debug("Processing %d results from search page %d for query '%s'",
                          len(search_results_list), page + 1, query_short)

            for result_index, search_item in enumerate(search_results_list):
                current_rank = start_index + result_index

                 # 1. Prepare Raw Result
                search_item['query'] = query
                search_item['search_page'] = page + 1
                search_item['approx_rank'] = current_rank
                raw_results_for_query.append(search_item.copy()) # Append a copy
                counters["raw_saved"] += 1

                # 2. Extract Text
                target_url = search_item.get('link')
                if not target_url:
                    logging.debug("Skipping extraction for item rank %d (no link) for query '%s'", current_rank, query_short)
                    # Still create an extracted record, but with null text
                    extracted_content = None
                else:
                    counters["urls_processed"] += 1
                    extracted_content = extract_main_text_requests(target_url, timeout=args.timeout)
                    if extracted_content is not None:
                        counters["extractions_success"] += 1
                    else:
                        logging.debug("Extraction failed or yielded no content for URL: %s", target_url)

                # 3. Prepare Extracted Result
                final_record = {
                    "query": query, "search_page": page + 1, "approx_rank": current_rank,
                    "url": target_url, "title": search_item.get('title', 'N/A'),
                    "extracted_text": extracted_content
                }
                extracted_results_for_query.append(final_record)

            if len(search_results_list) < args.num_results:
                logging.info("Received fewer results (%d) than requested (%d) for query '%s', page %d; stopping page fetch.",
                             len(search_results_list), args.num_results, query_short, page + 1)
                stop_fetching_pages = True
        else:
            logging.info("No valid URLs found in results for query '%s', page %d. Stopping page fetch.", query_short, page + 1)
            stop_fetching_pages = True

    logging.info("Finished pages for query '%s'. Raw results saved: %d. URLs processed: %d (%d successful).", query_short, counters["raw_saved"], counters["urls_processed"], counters["extractions_success"])
    return counters, raw_results_for_query, extracted_results_for_query


def run_process(args: argparse.Namespace):
    """Coordinates the fetching, extraction, and saving process."""
    try:
        config.validate_config()
        logging.info("API Key and CSE ID loaded successfully.")
    except ValueError as e:
        logging.error("%s. Exiting.", e)
        return

    queries = read_queries_from_vifactcheck(args.query_column)
    if queries is None: # Error reading file/column
        logging.error("Failed to read queries. Exiting.")
        return
    if not queries: # File/column ok, but no queries found
        logging.info("No queries to process. Exiting.")
        return
    
    # --- Prepare Output Directories ---
    try:
        # Raw results directory
        raw_base_path = args.search_output_base
        raw_output_dir = os.path.dirname(raw_base_path) or '.' # Use current dir if no path specified
        raw_base_name = os.path.basename(raw_base_path)
        raw_batch_dir = os.path.join(raw_output_dir, f"{raw_base_name}_batches")
        os.makedirs(raw_batch_dir, exist_ok=True)
        logging.info("Raw search batch results -> %s/", raw_batch_dir)

        # Extracted results directory
        extracted_base_path = args.extracted_output_base
        extracted_output_dir = os.path.dirname(extracted_base_path) or '.'
        extracted_base_name = os.path.basename(extracted_base_path)
        extracted_batch_dir = os.path.join(extracted_output_dir, f"{extracted_base_name}_batches")
        os.makedirs(extracted_batch_dir, exist_ok=True)
        logging.info("Extracted text batch results -> %s/", extracted_batch_dir)

    except OSError as e:
        logging.error("Failed to create output directories: %s. Exiting.", e)
        return

    # --- Initialize Total Counters ---
    total_counters = {"raw_saved": 0, "urls_processed": 0, "extractions_success": 0}
    actual_lines_written = {"raw": 0, "extracted": 0} # Track successful writes
    global_query_index = 0

    try:
        # --- Setup Batch Processing ---
        num_queries = len(queries)
        batch_size = args.batch_size
        num_batches = math.ceil(num_queries / batch_size)
        query_batches = batch_generator(queries, batch_size)

        logging.info("Processing %d queries in %d batches of size %d.", num_queries, num_batches, batch_size)
        
        # --- Process Batches with Progress Bar ---
        batch_iterator = tqdm(query_batches, total=num_batches, desc="Processing Batches", unit="batch")
        for batch_index, query_batch in enumerate(batch_iterator):
            batch_start_time = time.time()
            batch_iterator.set_postfix_str(f"Batch {batch_index + 1}/{num_batches}")

            # Lists to hold results for the current batch
            current_batch_raw_results: List[Dict] = []
            current_batch_extracted_results: List[Dict] = []

            # --- Process Queries within the Current Batch ---
            for query_in_batch_index, current_query in enumerate(query_batch):
                # --- Apply Delay ---
                if global_query_index > 0 and args.delay > 0:
                    logging.debug("Waiting %.2f seconds before query %d (%d in batch)...", args.delay, global_query_index + 1, query_in_batch_index + 1)
                    time.sleep(args.delay)

                # --- Process this Query (returns counters and results) ---
                query_counters, raw_res_list, extracted_res_list = _process_single_query(
                    current_query, args
                )

                # --- Aggregate Counters ---
                for key in total_counters:
                    total_counters[key] += query_counters.get(key, 0)

                # --- Collect Results for Batch ---
                current_batch_raw_results.extend(raw_res_list)
                current_batch_extracted_results.extend(extracted_res_list)
                global_query_index += 1

            # --- Batch Complete: Save Collected Results ---
            batch_end_time = time.time()
            logging.info("Batch %d completed processing %d queries in %.2f seconds.", batch_index + 1, len(query_batch), batch_end_time - batch_start_time)
            
            # Define batch filenames
            raw_batch_filename = os.path.join(raw_batch_dir, f"{raw_base_name}_{batch_index + 1}.jsonl")
            extracted_batch_filename = os.path.join(extracted_batch_dir, f"{extracted_base_name}_{batch_index + 1}.jsonl")

            logging.info("Saving batch %d results: Raw -> %s, Extracted -> %s", batch_index + 1, raw_batch_filename, extracted_batch_filename)

            # Save Raw Results for the batch
            raw_saved_count = 0
            try:
                with open(raw_batch_filename, 'w', encoding='utf-8') as batch_raw_outfile:
                    for record in current_batch_raw_results:
                        if save_jsonl_record(batch_raw_outfile, record):
                            raw_saved_count += 1
                actual_lines_written["raw"] += raw_saved_count
            except IOError as e:
                logging.error("Failed to open/write raw batch file %s: %s", raw_batch_filename, e)
            
            # Save Extracted Results for the batch
            extracted_saved_count = 0
            try:
                with open(extracted_batch_filename, 'w', encoding='utf-8') as batch_extracted_outfile:
                    for record in current_batch_extracted_results:
                         if save_jsonl_record(batch_extracted_outfile, record):
                            extracted_saved_count += 1
                actual_lines_written["extracted"] += extracted_saved_count
            except IOError as e:
                logging.error("Failed to open/write extracted batch file %s: %s", extracted_batch_filename, e)

            logging.info("Finished saving for batch %d. Raw saved: %.2f, Extracted saved: %.2f", batch_index + 1, raw_saved_count/len(current_batch_raw_results), extracted_saved_count/len(current_batch_extracted_results))

    except Exception as e:
        logging.error("An unexpected fatal error occurred during main execution: %s", e, exc_info=True)

    # --- Final Summary ---
    print("\n--- Processing Summary ---")
    print(f"Input Source: ViFactCheck Dataset (Column: '{args.query_column}')")
    print(f"Processed {global_query_index} out of {len(queries)} total queries.")
    print(f"Batch size: {args.batch_size}")
    # Counters reflect totals based on _process_single_query returns
    print(f"Total raw search results generated: {total_counters['raw_saved']}")
    print(f"Total URLs processed for extraction: {total_counters['urls_processed']}")
    print(f"Total successful text extractions: {total_counters['extractions_success']}")
    # Report actual lines written across all batches
    print(f"Total raw result lines successfully written to batch files: {actual_lines_written['raw']}")
    print(f"Total extracted text lines successfully written to batch files: {actual_lines_written['extracted']}")
    print(f"Raw search batch files saved in: {raw_batch_dir}/")
    print(f"Extracted text batch files saved in: {extracted_batch_dir}/")
    print("------------------------\n")


if __name__ == "__main__":
    args = parse_arguments()
    run_process(args)
    logging.info("Script finished.")
