# config.py
# -*- coding: utf-8 -*-

"""
Configuration settings, constants, and environment variable loading.
"""

import os
# import logging
from dotenv import load_dotenv

# --- Load Environment Variables ---
load_dotenv()
API_KEY = os.getenv("GOOGLE_CUSTOM_SEARCH_API_KEY")
CSE_ID = os.getenv("GOOGLE_CSE_ID")

# --- Constants ---

# Google CSE API
GOOGLE_API_URL = "https://www.googleapis.com/customsearch/v1"
DEFAULT_CSE_NUM_RESULTS = 10 # Max 10 per request for CSE API
DEFAULT_REQUEST_DELAY = 1.0 # Delay between CSE API calls (seconds)
CSE_API_TIMEOUT = 15 # Timeout for CSE API requests

# Text Extraction
DEFAULT_EXTRACTION_TIMEOUT = 15 # Timeout for fetching each URL (seconds)
EXTRACTION_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

# Folder Base Naming Defaults
DEFAULT_EXTRACTED_OUTPUT_BASE = "data/retrieval_results/raw_search"
DEFAULT_SEARCH_OUTPUT_BASE = "data/retrieval_results/extracted_text"

# --- Validation (Optional but Recommended) ---
def validate_config():
    """Basic check if essential API keys are loaded."""
    if not API_KEY:
        raise ValueError("Configuration Error: GOOGLE_API_KEY not found in environment/.env")
    if not CSE_ID:
        raise ValueError("Configuration Error: GOOGLE_CSE_ID not found in environment/.env")

# You could call validate_config() here if you want it checked on import,
# or call it explicitly from main_script.py
# validate_config()
