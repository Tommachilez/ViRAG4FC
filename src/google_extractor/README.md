# Google Search & Content Extractor

This folder contains the following Python scripts:

- `data_handler.py`
- `google_cse.py`
- `content_extractor.py`
- `document_retrieval.py`
- `config.py`

## How to Use 🚀

Run the script from your terminal using the command line.

- `--query-column` (Required): The name of the column in the dataset containing the search queries.

- `-p`, `--pages`: Number of Google Search result pages to fetch for each query. (Default: 1)

- `-n`, `--num-results`: Number of results to fetch per page (max 10). (Default: 10)

- `--batch-size`: Number of queries to process before saving a batch of results. (Default: 10)

- `--delay`: Seconds to wait between API calls for different queries. (Default: 1.0)

- `--timeout`: Seconds to wait for a webpage to respond during text extraction. (Default: 15)

- `--site-search`: Restrict search to a specific domain (e.g., nytimes.com).

- `--search-output-base`: Base path and prefix for raw search result files.

- `--extracted-output-base`: Base path and prefix for extracted text files.

## Example Run Commands

```bash
python vifactcheck_query.py --query-column "claim" -n 5 --batch-size 50
```

This command processes queries from the `claim` column of the dataset. It fetches the top 5 results (`-n 5`) from the first page (`-p 1`) for each query. It processes 50 queries at a time (`--batch-size 50`) and saves the output to the default `output/` directory.

## Output Format

### Raw Search Results

These files contain the unprocessed results returned directly by the Google CSE API.

- **Location**: `output/raw_search_batches/` (based on default settings)

- **File Naming**: `raw_1.jsonl`, `raw_2.jsonl`, etc.

```json
{"title": "Example News Article Title", "link": "https://example.com/news/article", "snippet": "A brief snippet describing the content of the linked page...", "query": "The original search query", "search_page": 1, "approx_rank": 1}
```

### Extracted Text Results

These files contain the key information and the main text content extracted from each URL.

- **Location**: `output/extracted_text_batches/` (based on default settings)

- **File Naming**: `text_1.jsonl`, `text_2.jsonl`, etc.

If text extraction fails for a URL, the `extracted_text` field will be `null`.

```json
{"query": "The original search query", "search_page": 1, "approx_rank": 1, "url": "https://example.com/news/article", "title": "Example News Article Title", "extracted_text": "This is the full main text content extracted from the webpage, with navigation, ads, and footers removed..."}
```
