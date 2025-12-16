import sys
import csv
import json
import string
import argparse
import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Set
from tqdm import tqdm
import py_vncorenlp
from underthesea import text_normalize

try:
    from pyserini.search.lucene import LuceneSearcher
except ImportError:
    print("Error: Pyserini not installed. Please run 'pip install pyserini faiss-cpu'", file=sys.stderr)
    sys.exit(1)

# ==========================================
# CONFIGURATION
# ==========================================

STOPWORD_WHITELIST = {
    # Negation
    "không", "chẳng", "chả", "chưa", "phi", "vô", "tránh", "đừng", "chớ",
    # Logic & Connection
    "và", "hoặc", "nhưng", "tuy", "nếu", "thì", "vì", "do", "bởi", "tại", "nên", 
    "rằng", "là", "của", "thuộc",
    # Prepositions
    "tại", "ở", "trong", "ngoài", "trên", "dưới", "giữa", "với", "về", "đến",
    # Question Words
    "ai", "gì", "nào", "đâu", "khi", "mấy", "bao_nhiêu", "thế_nào", "sao",
    # Passive/Active Markers
    "bị", "được", "do", "bởi"
}

# ==========================================
# UTILS & TOKENIZER (Reference Flow)
# ==========================================

class VietnameseQueryProcessor:
    _vncorenlp = None
    _vncorenlp_path = None
    punctuation = set(string.punctuation)

    def __init__(self, vncorenlp_path: str):
        if not vncorenlp_path or not Path(vncorenlp_path).exists():
            print(f"Error: VnCoreNLP path not found at '{vncorenlp_path}'", file=sys.stderr)
            sys.exit(1)

        VietnameseQueryProcessor._vncorenlp_path = vncorenlp_path
        self.get_vncorenlp()  # Initialize

    @classmethod
    def get_vncorenlp(cls):
        """Initializes and returns the py_vncorenlp singleton instance."""
        if cls._vncorenlp is None:
            save_dir = str(cls._vncorenlp_path)
            try:
                # Suppress stdout momentarily if needed
                cls._vncorenlp = py_vncorenlp.VnCoreNLP(
                    save_dir=save_dir,
                    annotators=["wseg"]
                )
            except Exception as e:
                print(f"Error initializing VnCoreNLP: {e}", file=sys.stderr)
                sys.exit(1)
        return cls._vncorenlp

    def process_query(self, query: str) -> List[str]:
        """
        Tokenizes and normalizes the query.
        Returns a LIST of tokens (retaining order and duplicates for BM25 TF calculation).
        """
        rdrsegmenter = self.get_vncorenlp()

        # 1. Lowercase and normalize
        try:
            query = text_normalize(query.lower())
        except Exception:
            query = query.lower()

        # 2. Segment words
        try:
            # VnCoreNLP returns list of sentences
            segmented_sents = rdrsegmenter.word_segment(query)
        except Exception as e:
            print(f"VnCoreNLP error: {e}", file=sys.stderr)
            segmented_sents = []

        # 3. Flatten list of sentences and split terms
        query_terms = [term for sent in segmented_sents for term in sent.split(' ')]

        # 4. Filter out punctuation and empty strings
        return list(filter(lambda x: x not in self.punctuation and x.strip(), query_terms))


# ==========================================
# LEXICAL FILTER LOGIC
# ==========================================
class LexicalFilter:
    def __init__(self, stopwords_path: str):
        self.stopwords = self._load_stopwords(stopwords_path)

    def _load_stopwords(self, path: str) -> Set[str]:
        final_stopwords = set()
        path_obj = Path(path)

        if not path_obj.exists():
            print(f"Warning: Stopwords file not found at {path}. No filtering will occur.")
            return set()

        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip().lower()
                if not word:
                    continue

                # Normalize delimiters: VnCoreNLP uses underscores (_) for compounds.
                # We replace dashes/spaces with underscores to ensure they match tokenizer output.
                token_style = word.replace(' ', '_').replace('-', '_')

                # Check Whitelist (Reference Logic)
                if token_style in STOPWORD_WHITELIST:
                    continue

                final_stopwords.add(token_style)

        print(f"Loaded {len(final_stopwords)} stopwords.")
        return final_stopwords

    def filter_tokens(self, tokens: List[str]) -> List[str]:
        """Filters tokens based on the loaded stopwords."""
        return [t for t in tokens if t not in self.stopwords]


# ==========================================
# MAIN MINING SCRIPT
# ==========================================

def load_documents(csv_path: str, doc_col: str, id_col: str):
    """Loads documents from CSV into a dictionary {id: text}."""
    documents = {}
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            # Validate columns
            if doc_col not in reader.fieldnames or id_col not in reader.fieldnames:
                print(f"Error: Columns '{doc_col}' or '{id_col}' not found in CSV.", file=sys.stderr)
                sys.exit(1)

            for row in reader:
                doc_id = row[id_col].strip()
                content = row[doc_col].strip()
                if doc_id and content:
                    documents[doc_id] = content
    except FileNotFoundError:
        print(f"Error: Document file {csv_path} not found.", file=sys.stderr)
        sys.exit(1)
    return documents

def main():
    parser = argparse.ArgumentParser(description="Pyserini BM25 Hard Negative Mining for Vietnamese Text")
    parser.add_argument("--doc_csv", type=str, required=True, help="Input CSV file with documents.")
    parser.add_argument("--query_jsonl", type=str, required=True, help="Input JSONL file with generated queries.")
    parser.add_argument("--stopwords_path", type=str, required=True, help="Path to Vietnamese stopwords file.")
    parser.add_argument("--vncorenlp_path", type=str, required=True, help="Path to VnCoreNLP directory.")
    parser.add_argument("--output_jsonl", type=str, required=True, help="Output JSONL file path.")
    parser.add_argument("--quota", type=int, default=None, help="Quota limit for processing input lines.")
    parser.add_argument("--doc_col", type=str, default="document", help="CSV column name for document content.")
    parser.add_argument("--id_col", type=str, default="id", help="CSV/JSONL key name for document ID.")
    parser.add_argument("--top_k", type=int, default=10, help="Number of hard negatives to mine per query.")

    args = parser.parse_args()

    # Define temp paths for Pyserini
    TEMP_CORPUS_DIR = "temp_pyserini_corpus"
    TEMP_INDEX_DIR = "temp_pyserini_index"

    # Clean up previous runs if they exist
    if os.path.exists(TEMP_CORPUS_DIR): shutil.rmtree(TEMP_CORPUS_DIR)
    if os.path.exists(TEMP_INDEX_DIR): shutil.rmtree(TEMP_INDEX_DIR)
    os.makedirs(TEMP_CORPUS_DIR, exist_ok=True)

    # 1. Initialize Processor and Filter
    print(">>> Initializing NLP components...")
    processor = VietnameseQueryProcessor(args.vncorenlp_path)
    lexical_filter = LexicalFilter(args.stopwords_path)

    # 2. Load and Prepare Documents for Pyserini
    print(f">>> Loading corpus from {args.doc_csv}...")
    doc_map = load_documents(args.doc_csv, args.doc_col, args.id_col)

    print(">>> Pre-processing corpus and creating Pyserini input files...")
    # Pyserini expects JSONL: {"id": "...", "contents": "..."}
    # We pre-segment "contents" with spaces so Pyserini's WhitespaceAnalyzer works well.
    with open(os.path.join(TEMP_CORPUS_DIR, "docs.jsonl"), 'w', encoding='utf-8') as f_corpus:
        for doc_id, text in tqdm(doc_map.items(), desc="Preprocessing Corpus"):
            raw_tokens = processor.process_query(text)
            filtered_tokens = lexical_filter.filter_tokens(raw_tokens)
            # Join tokens with space for Pyserini to digest
            processed_text = " ".join(filtered_tokens)

            json_line = json.dumps({"id": doc_id, "contents": processed_text}, ensure_ascii=False)
            f_corpus.write(json_line + "\n")

    # 3. Build Lucene Index via Command Line
    print(">>> Building Pyserini Lucene index...")
    # We use 'JsonCollection' and generate the index
    # We use WhitespaceAnalyzer by passing -analyzeWithWhitespace (if supported) or reliance on default.
    # Standard Pyserini practice for pre-tokenized text is to rely on simple whitespace splitting.
    cmd = [
        "python", "-m", "pyserini.index.lucene",
        "--collection", "JsonCollection",
        "--input", TEMP_CORPUS_DIR,
        "--index", TEMP_INDEX_DIR,
        "--generator", "DefaultLuceneDocumentGenerator",
        "--threads", "1",
        "--storePositions", "--storeDocvectors", "--storeRaw",
        "--pretokenized"
    ]

    # Execute indexing
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        print("Error building index:", result.stderr)
        sys.exit(1)
    else:
        print(">>> Index build successful.")

    # 4. Initialize Searcher
    print(">>> Initializing Searcher...")
    searcher = LuceneSearcher(TEMP_INDEX_DIR)
    # BM25 parameters (k1=1.2, b=0.75 is default, but you can tune)
    searcher.set_bm25(k1=1.2, b=0.75)

    # 5. Mine Negatives
    print(f">>> Processing queries from {args.query_jsonl}...")
    count = 0
    quota = args.quota if args.quota else float('inf')

    with open(args.query_jsonl, 'r', encoding='utf-8') as f_in, \
         open(args.output_jsonl, 'w', encoding='utf-8') as f_out:

        for line in tqdm(f_in, desc="Mining"):
            if count >= quota:
                break

            try:
                data = json.loads(line)
                pos_doc_id = str(data.get(args.id_col))

                if pos_doc_id not in doc_map:
                    continue

                generated_queries = data.get("generated_queries", [])
                if not generated_queries:
                    continue

                for q_obj in generated_queries:
                    query_text = q_obj.get("query")
                    if not query_text:
                        continue

                    # Pre-process query exactly like documents
                    raw_q_tokens = processor.process_query(query_text)
                    filtered_q_tokens = lexical_filter.filter_tokens(raw_q_tokens)
                    processed_query = " ".join(filtered_q_tokens)

                    if not processed_query.strip():
                        continue

                    # Retrieve (Top K + Buffer)
                    # We need extra candidates because we must filter out the positive doc
                    hits = searcher.search(processed_query, k=args.top_k + 10)

                    hard_negatives = []
                    for hit in hits:
                        # Pyserini returns docid as string
                        candidate_id = hit.docid

                        if candidate_id == pos_doc_id:
                            continue

                        # Retrieve original raw text from our memory map
                        candidate_text = doc_map.get(candidate_id)
                        if candidate_text:
                            hard_negatives.append(candidate_text)

                        if len(hard_negatives) >= args.top_k:
                            break

                    output_obj = {
                        "query": query_text,
                        "pos": [doc_map[pos_doc_id]],
                        "neg": hard_negatives
                    }
                    f_out.write(json.dumps(output_obj, ensure_ascii=False) + "\n")

                count += 1

            except json.JSONDecodeError:
                continue
            except Exception as e:
                print(f"Error processing line: {e}", file=sys.stderr)
                continue

    # Cleanup (Optional)
    print(">>> Cleaning up temporary indices...")
    shutil.rmtree(TEMP_CORPUS_DIR)
    shutil.rmtree(TEMP_INDEX_DIR)

    print(f">>> Done. Processed {count} source lines. Output saved to {args.output_jsonl}")

if __name__ == "__main__":
    main()
