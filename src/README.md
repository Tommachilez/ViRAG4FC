# Google Search & Content Extractor

## How to Use 🚀

The main script is `document_retrieval.py`, which supports two primary execution modes: **Single Query Mode** and **Batch File Mode**.

Run the script from your terminal using the command line with the following arguments:

|Argument|Description|Required?|Batch Mode Only?|Default|
|--------|-----------|---------|----------------|--------|
|"-q, --query"|A single search query to process. Activates Single Query Mode.|Mutually Exclusive|No|N/A|
|"-i, --input-file"|Input CSV/TSV file containing queries. Activates Batch File Mode.|Mutually Exclusive|No|N/A|
|--query-column|Column name in the dataset containing the search queries.|Required for Batch Mode|Yes|N/A|
|--num-queries|Number of queries from the input file to process.|No|Yes|All|
|"-p, --pages"|Number of Google Search result pages to fetch for each query.|No|No|1|
|"-n, --num-results"|Number of results to fetch per page (max 10).|No|No|10|
|"-d, --delay"|Seconds to wait between CSE API calls for different queries.|No|No|1.0|
|"-t, --timeout"|Seconds to wait for a webpage to respond during text extraction.|No|No|15|
|--site-search|"Restrict search to a specific domain (e.g., nytimes.com)."|No|No|N/A|
|--file-type|"Restrict search to a specific file type (e.g., pdf, docx)."|No|No|N/A|
|--batch-size|Number of queries to process before saving a batch of results.|No|Yes|10|
|--search-output-base|Base path/prefix for raw search result files.|No|No|output/raw_search_batches/raw|
|--extracted-output-base|Base path/prefix for extracted text files.|No|No|output/extracted_text_batches/text|

## Example Run Commands

### 1. Batch File Mode

Processes queries from a file (`queries.csv`), using the column named `claim`. It fetches the top 5 results (`-n 5`) from the first page (`-p 1`) for each query and processes 50 queries at a time (`--batch-size 50`).

```bash
python document_retrieval.py --input-file "queries.csv" --query-column "claim" -n 5 --batch-size 50
```

### 2. Single Query Mode

Processes a single query directly from the command line, fetches the top 10 results from the first 2 pages (`-p 2`), and restricts the search to a specific site. The output files will have a timestamp for uniqueness.

```bash
python document_retrieval.py -q "Benefits of Python" -p 2 --site-search "realpython.com" --search-output-base "output/single/raw" --extracted-output-base "output/single/text"
```

## Output Format

The script creates timestamped directories for batches in Batch Mode.

### Raw Search Results

These files contain the unprocessed results returned directly by the Google CSE API.

- Location (Batch Mode): A timestamped directory like `output/raw_search_batches/YYYYMMDD-HHMMSS_raw_batches/`

- Location (Single Mode): A unique, timestamped file in the directory specified by `--search-output-base`.

- File Naming (Batch Mode): `raw_1.jsonl`, `raw_2.jsonl`, etc.

```json
{"title": "Example News Article Title", "link": "https://example.com/news/article", "snippet": "A brief snippet describing the content of the linked page...", "query": "The original search query", "search_page": 1, "approx_rank": 1}
```

### Extracted Text Results

These files contain the key information and the main text content extracted from each URL.

- Location (Batch Mode): A timestamped directory like `output/extracted_text_batches/YYYYMMDD-HHMMSS_text_batches/`

- Location (Single Mode): A unique, timestamped file in the directory specified by `--extracted-output-base`.

- File Naming (Batch Mode): `text_1.jsonl`, `text_2.jsonl`, etc.

If text extraction fails for a URL, the `extracted_text` field will be `null`.
