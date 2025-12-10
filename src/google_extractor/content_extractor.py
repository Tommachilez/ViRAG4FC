# text_extractor.py
# -*- coding: utf-8 -*-

"""
Functions for extracting main text content from URLs.
"""

import logging
from typing import Optional

import requests
import trafilatura

from utils.config import EXTRACTION_HEADERS, DEFAULT_EXTRACTION_TIMEOUT


try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None
    logging.warning("PyMuPDF is not installed. PDF extraction will not be available. Run 'pip install PyMuPDF'.")


def _extract_text_from_pdf_content(pdf_content: bytes, url: str) -> Optional[str]:
    """Extracts text from PDF byte content using PyMuPDF."""
    if not fitz:
        logging.error("Cannot process PDF from '%s', PyMuPDF (fitz) is not installed.", url)
        return None
    
    try:
        # Open PDF from memory stream
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        all_text = [page.get_text() for page in doc]
        
        full_text = "\n".join(all_text)
        logging.debug("[PDF] Successfully extracted text from %s", url)
        return full_text.strip()
    except Exception as e:
        logging.error("[PDF] Failed to extract text from %s due to: %s", url, e)
        return None


def extract_main_text_requests(url: str, timeout: int = DEFAULT_EXTRACTION_TIMEOUT) -> Optional[str]:
    """
    Fetches content from URL, determines type, and extracts main text.
    Handles HTML using trafilatura and PDF using PyMuPDF.
    
    Uses lazy % formatting for logging. Returns text string or None on failure.
    """
    logging.debug("[Requests] Attempting text extraction from: %s", url)
    try:
        response = requests.get(url, headers=EXTRACTION_HEADERS, timeout=timeout, allow_redirects=True)
        response.raise_for_status() # Check for HTTP errors (4xx, 5xx)
        content_type = response.headers.get('content-type', '').lower()

        # Route to PDF extractor if content type matches
        if 'application/pdf' in content_type:
            logging.info("Detected PDF content type for URL: %s", url)
            return _extract_text_from_pdf_content(response.content, url)

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
