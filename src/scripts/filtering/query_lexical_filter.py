import sys
import json
import csv
import string
import argparse
from pathlib import Path
from typing import Set, Dict
import py_vncorenlp
from underthesea import text_normalize
from tqdm import tqdm

# ==========================================
# STOPWORD WHITELIST
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
# UTILS & TOKENIZER
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
            # Suppress stdout momentarily if needed, strictly keeping logs minimal
            try:
                cls._vncorenlp = py_vncorenlp.VnCoreNLP(
                    save_dir=save_dir,
                    annotators=["wseg"]
                )
            except Exception as e:
                print(f"Error initializing VnCoreNLP: {e}", file=sys.stderr)
                sys.exit(1)
        return cls._vncorenlp

    def process_query(self, query: str) -> Set[str]:
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
        return set(filter(lambda x: x not in self.punctuation and x.strip(), query_terms))


# ==========================================
# LEXICAL FILTER LOGIC
# ==========================================
class LexicalFilter:
    def __init__(self, processor: VietnameseQueryProcessor, stopwords_path: str, use_whitelist: bool = False):
        self.processor = processor
        self.use_whitelist = use_whitelist
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

                # Check Whitelist
                if self.use_whitelist and token_style in STOPWORD_WHITELIST:
                    continue

                final_stopwords.add(token_style)

        print(f"Loaded {len(final_stopwords)} stopwords (Whitelist enabled: {self.use_whitelist}).")
        return final_stopwords

    def calculate_overlap(self, query_tokens: Set[str], doc_tokens: Set[str]) -> float:
        """
        Calculates overlap percentage of NON-STOPWORD tokens.
        """
        meaningful_query = {t for t in query_tokens if t not in self.stopwords}

        if not meaningful_query:
            return 0.0

        meaningful_doc = {t for t in doc_tokens if t not in self.stopwords}
        intersection = meaningful_query.intersection(meaningful_doc)

        return len(intersection) / len(meaningful_query)


def load_documents(csv_path: str) -> Dict[str, str]:
    docs = {}
    path_obj = Path(csv_path)
    if not path_obj.exists():
        print(f"Error: CSV file not found at {csv_path}", file=sys.stderr)
        sys.exit(1)

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if 'id' in row and 'document' in row:
                docs[row['id']] = row['document']
    print(f"Loaded {len(docs)} documents.")
    return docs


def count_lines(filepath):
    """Quickly count lines in a file for the progress bar."""
    print("Counting lines in JSONL file...")
    with open(filepath, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f)


# ==========================================
# MAIN EXECUTION
# ==========================================
def parse_arguments():
    parser = argparse.ArgumentParser(description="Filter generated queries based on lexical overlap.")

    parser.add_argument("--vncorenlp", required=True, help="Path to VnCoreNLP folder")
    parser.add_argument("--csv", required=True, help="Path to input CSV (id, document)")
    parser.add_argument("--jsonl", required=True, help="Path to input JSONL (generated queries)")
    parser.add_argument("--stopwords", required=True, help="Path to stopwords TXT")
    parser.add_argument("--output_dir", required=True, help="Directory to store output JSONL files")
    parser.add_argument("--threshold", type=float, default=0.5, help="Overlap threshold (0.0-1.0)")
    parser.add_argument("--quota", type=int, default=None, help="Max number of documents to process. Default: Process all.")
    parser.add_argument("--enable_whitelist", action="store_true", help="If set, prevents specific important stopwords (negations, logic) from being filtered out.")

    return parser.parse_args()

def main():
    args = parse_arguments()

    # 1. Initialize Components
    processor = VietnameseQueryProcessor(args.vncorenlp)
    lexical_filter = LexicalFilter(processor, args.stopwords, use_whitelist=args.enable_whitelist)
    documents = load_documents(args.csv)

    # 2. Prepare Output Directory
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True) # Create folder if not exists
    print(f"Output directory set to: {out_dir.resolve()}")

    files = {
        "keyword": open(out_dir / "keyword.jsonl", "w", encoding="utf-8"),
        "natural": open(out_dir / "natural.jsonl", "w", encoding="utf-8"),
        "semantic": open(out_dir / "semantic.jsonl", "w", encoding="utf-8"),
        "filtered_source": open(out_dir / "filtered_source.jsonl", "w", encoding="utf-8")
    }

    doc_token_cache = {}

    # 3. Determine work volume
    try:
        total_file_lines = count_lines(args.jsonl)
    except FileNotFoundError:
        print(f"Error: JSONL file not found at {args.jsonl}", file=sys.stderr)
        return

    # If quota is set and smaller than file size, use quota as total
    if args.quota is not None:
        total_to_process = min(args.quota, total_file_lines)
    else:
        total_to_process = total_file_lines

    print(f"Processing {total_to_process} entries with threshold {args.threshold}...")

    # 4. Process Loop
    count_processed = 0
    debug_printed = False # Flag to print debug info only once

    try:
        with open(args.jsonl, 'r', encoding='utf-8') as f:
            pbar = tqdm(total=total_to_process, desc="Filtering", unit="docs")

            for line in f:
                if args.quota is not None and count_processed >= args.quota: break

                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # FORCE ID TO STRING
                doc_id = str(entry.get('id')).strip()
                queries = entry.get('generated_queries', [])

                # --- DEBUG BLOCK START ---
                if not debug_printed:
                    print(f"\n[DEBUG] First Entry ID in JSON: '{doc_id}'")
                    if doc_id in documents:
                        print(f"[DEBUG] ID '{doc_id}' FOUND in CSV documents.")
                    else:
                        print(f"[DEBUG] ID '{doc_id}' NOT FOUND in CSV. (First 5 keys in CSV: {list(documents.keys())[:5]})")
                    print(f"[DEBUG] Queries structure: {json.dumps(queries, ensure_ascii=False)[:200]}...")
                    debug_printed = True
                # --- DEBUG BLOCK END ---

                if doc_id not in documents:
                    # Skip if doc not found
                    continue

                if doc_id not in doc_token_cache:
                    doc_text = documents[doc_id]
                    doc_token_cache[doc_id] = processor.process_query(doc_text)
                doc_tokens = doc_token_cache[doc_id]

                # Store valid queries for this specific document to write to filtered_source.jsonl
                valid_queries_for_this_doc = []

                for q in queries:
                    q_type = q.get('type', '').lower()
                    q_text = q.get('query', '')
                    is_valid = False # Track if this specific query passed checks

                    if not q_text: continue

                    if 'semantic' in q_type:
                        # Semantic queries usually pass without lexical overlap check
                        is_valid = True
                        json.dump({"id": doc_id, "query": q}, files["semantic"], ensure_ascii=False)
                        files["semantic"].write("\n")

                    elif 'keyword' in q_type or 'natural' in q_type:
                        q_tokens = processor.process_query(q_text)
                        overlap = lexical_filter.calculate_overlap(q_tokens, doc_tokens)

                        # DEBUG OVERLAP
                        if count_processed < 5:
                            print(f"[DEBUG] Type: {q_type} | Overlap: {overlap:.2f} | Threshold: {args.threshold}")

                        target_file = files["keyword"] if 'keyword' in q_type else files["natural"]
                        json.dump({"id": doc_id, "query": q, "score": round(overlap, 4)}, target_file, ensure_ascii=False)
                        target_file.write("\n")

                        if overlap >= args.threshold:
                            is_valid = True

                    if is_valid:
                        valid_queries_for_this_doc.append(q)

                if valid_queries_for_this_doc:
                    source_entry = {
                        "id": doc_id,
                        "generated_queries": valid_queries_for_this_doc
                    }
                    json.dump(source_entry, files["filtered_source"], ensure_ascii=False)
                    files["filtered_source"].write("\n")

                count_processed += 1
                pbar.update(1)
            pbar.close()

    except Exception as e:
        print(f"\nUnexpected error: {e}")
    finally:
        for f in files.values():
            f.close()
        print(f"\nDone. Results saved to {out_dir}")

if __name__ == "__main__":
    main()
