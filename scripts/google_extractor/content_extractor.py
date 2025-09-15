# text_extractor.py
# -*- coding: utf-8 -*-

"""
Functions for extracting main text content from URLs.
Includes a requests-based version and a Playwright-based version for dynamic content.
"""

import logging
from typing import Optional

import requests
import trafilatura

from config import EXTRACTION_HEADERS, DEFAULT_EXTRACTION_TIMEOUT


def extract_main_text_requests(url: str, timeout: int = DEFAULT_EXTRACTION_TIMEOUT) -> Optional[str]:
    """
    Fetches content from URL, extracts main text using trafilatura.
    Uses lazy % formatting for logging. Returns text string or None on failure.
    """
    logging.debug("[Requests] Attempting text extraction from: %s", url)
    try:
        response = requests.get(url, headers=EXTRACTION_HEADERS, timeout=timeout, allow_redirects=True)
        response.raise_for_status() # Check for HTTP errors (4xx, 5xx)
        content_type = response.headers.get('content-type', '').lower()

        # Check if content type seems appropriate before parsing
        if 'html' not in content_type and 'xml' not in content_type:
            logging.warning("[Requests] Content type '%s' for URL %s not HTML/XML. Skipping extraction.", content_type, url)
            return None

        html_content = response.text
        if not html_content:
            logging.warning("[Requests] No HTML/text content retrieved from %s", url)
            return None

        # Use trafilatura for extraction
        # Consider adding error_recovery=True for more resilience if needed
        main_text = trafilatura.extract(
            html_content,
            include_comments=False,
            include_tables=True, # Adjust as needed
            output_format='txt'
            # target_language='en' # Optional: specify if known
        )

        if main_text:
            logging.debug("[Requests] Successfully extracted text from: %s", url)
            return main_text.strip() # Remove leading/trailing whitespace

        # It's not necessarily an error if a page has no extractable main text
        logging.info("[Requests] Trafilatura found no main text in: %s", url)
        return None

    except requests.exceptions.Timeout:
        logging.error("[Requests] Timeout occurred while fetching URL for extraction: %s", url)
        return None
    except requests.exceptions.TooManyRedirects:
        logging.error("[Requests] Too many redirects for URL: %s", url)
        return None
    except requests.exceptions.RequestException as e:
        # Log other request errors but allow the main script to continue
        logging.error("[Requests] Extraction error fetching %s: %s", url, e)
        return None
    except Exception as e:
        # Catch potential trafilatura or other unexpected errors
        logging.error(
            "[Requests] Unexpected error during text extraction for %s: %s",
            url,
            e,
            exc_info=False) # Set exc_info=True for full traceback if needed
        return None
