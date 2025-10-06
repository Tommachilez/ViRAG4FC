# Google Search.py
# -*- coding: utf-8 -*-

"""
Functions for interacting with the Google Custom Search Engine (CSE) API.
"""

import logging
import json
from typing import List, Dict, Any, Optional

import requests

# Import constants from config module
from config import GOOGLE_API_URL, CSE_API_TIMEOUT

def search_google_cse(
    query: str,
    api_key: str,
    cse_id: str,
    num_results: int,
    include_file_type: Optional[str] = None,
    exclude_file_type: Optional[str] = None,
    start_index: int = 1,
    **kwargs: Any
) -> Optional[Dict[str, Any]]:
    """
    Performs a search query using the Google Custom Search API.
    Uses lazy % formatting for logging.
    """
    # Basic validation handled in config, but check args passed specifically here if needed
    if not api_key or not cse_id:
        logging.error("API Key or CSE ID missing in call to search_google_cse.")
        # Or raise ValueError - depends on desired handling in caller
        return None

    # num_results validation happens in main_script based on default/args

    params = {
        'key': api_key, 'cx': cse_id, 'q': query,
        'num': num_results, 'start': start_index, **kwargs
    }

    if include_file_type:
        params['q'] = f"{query} filetype:{include_file_type}"
        logging.info("Modified query with file type inclusion: '%s'", params['q'])
    elif exclude_file_type:
        params['q'] = f"{query} -filetype:{exclude_file_type}"
        logging.info("Modified query with file type exclusion: '%s'", params['q'])

    logging.info("Sending request to Google CSE API for query: '%s' (start: %d, num: %d)",
                 query, start_index, num_results)

    try:
        response = requests.get(GOOGLE_API_URL, params=params, timeout=CSE_API_TIMEOUT)
        response.raise_for_status() # Check for 4xx/5xx errors
        results = response.json()
        # Check for API-level errors within the JSON response
        if 'error' in results:
            error_details = results['error']
            logging.error("Google API Error: Code %s - %s",
                          error_details.get('code'), error_details.get('message'))
            return None # Return None to indicate API error, distinct from network error
        return results
    except requests.exceptions.Timeout:
        logging.error("Google CSE API Request timed out for query: %s", query)
        return None # Indicate timeout
    except requests.exceptions.RequestException as e:
        logging.error("Network error during CSE API request for query '%s': %s", query, e)
        raise # Re-raise network errors to be handled by the caller (main_script)
    except json.JSONDecodeError:
        logging.error("Failed to decode JSON response from CSE API for query: %s", query)
        return None # Indicate bad response format
   

def process_search_results(results_json: Optional[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Extracts key information (title, link, snippet) from the CSE API response.
    Filters for valid links.
    """
    if not results_json or 'items' not in results_json:
        logging.debug("No search results items found in CSE API response or response is invalid.")
        return []

    processed = []
    for item in results_json['items']:
        link = item.get('link')
        # Ensure link exists and looks like a standard URL
        if link and link.startswith(('http://', 'https://')):
            processed.append({
            'title': item.get('title', 'N/A'),
            'link': link,
            'snippet': item.get('snippet', 'N/A')
            })
        else:
            logging.warning("Skipping result with invalid/missing link: Title '%s'", item.get('title', 'N/A'))
    return processed
