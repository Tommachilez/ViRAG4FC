import os
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
# STOPWORD WHITELIST (Reused for consistency)
# ==========================================
STOPWORD_WHITELIST = {
    "không", "chẳng", "chả", "chưa", "phi", "vô", "tránh", "đừng", "chớ",
    "và", "hoặc", "nhưng", "tuy", "nếu", "thì", "vì", "do", "bởi", "tại", "nên", 
    "rằng", "là", "của", "thuộc",
    "tại", "ở", "trong", "ngoài", "trên", "dưới", "giữa", "với", "về", "đến",
    "ai", "gì", "nào", "đâu", "khi", "mấy", "bao_nhiêu", "thế_nào", "sao",
    "bị", "được", "do", "bởi"
}

# ==========================================
# TEXT PROCESSOR & FILTER
# ==========================================
class VietnameseQueryProcessor:
    _vncorenlp = None
    _vncorenlp_path = None
    punctuation = set(string.punctuation)

    def __init__(self, vncorenlp_path: str):
        if not os.path.exists(vncorenlp_path):
            raise FileNotFoundError(f"VnCoreNLP not found at {vncorenlp_path}")
        VietnameseQueryProcessor._vncorenlp_path = vncorenlp_path
        self.get_vncorenlp()

    @classmethod
    def get_vncorenlp(cls):
        if cls._vncorenlp is None:
            # Initialize strictly with word segmentation
            cls._vncorenlp = py_vncorenlp.VnCoreNLP(
                save_dir=str(cls._vncorenlp_path),
                annotators=["wseg"]
            )
        return cls._vncorenlp

    def process_text(self, text: str) -> Set[str]:
        """Segments and normalizes text into a set of tokens."""
        if not text:
            return set()

        rdrsegmenter = self.get_vncorenlp()

        # 1. Normalize
        try:
            text = text_normalize(text.lower())
        except Exception:
            text = text.lower()

        # 2. Segment
        try:
            segmented_sents = rdrsegmenter.word_segment(text)
        except Exception:
            return set()

        # 3. Flatten & Filter
        tokens = [term for sent in segmented_sents for term in sent.split(' ')]
        return set(filter(lambda x: x not in self.punctuation and x.strip(), tokens))

class LexicalCalculator:
    def __init__(self, processor: VietnameseQueryProcessor, stopwords_path: str):
        self.processor = processor
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
                if not word: continue

                # Ensure format matches tokenizer output (underscores for spaces)
                token_style = word.replace(' ', '_').replace('-', '_')

                # if token_style in STOPWORD_WHITELIST:
                #     continue
                final_stopwords.add(token_style)

        print(f"Loaded {len(final_stopwords)} stopwords.")
        return final_stopwords

    def calculate_overlap(self, query_tokens: Set[str], doc_tokens: Set[str]) -> float:
        """
        Overlap = (Intersection of Non-Stopwords) / (Count of Query Non-Stopwords)
        """
        meaningful_query = {t for t in query_tokens if t not in self.stopwords}

        if not meaningful_query:
            return 0.0

        meaningful_doc = {t for t in doc_tokens if t not in self.stopwords}
        intersection = meaningful_query.intersection(meaningful_doc)

        return len(intersection) / len(meaningful_query)

# ==========================================
# DATA LOADING UTILS
# ==========================================
def load_document_map(csv_path: str) -> Dict[str, str]:
    """Loads Doc ID -> Document Text mapping."""
    print(f"Loading documents from {csv_path}...")
    docs = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if 'doc_id' in row and 'document' in row:
                docs[row['doc_id'].strip()] = row['document']
    return docs

def load_query_map(csv_path: str) -> Dict[str, str]:
    """Loads Query ID -> Query Text mapping."""
    print(f"Loading queries from {csv_path}...")
    queries = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if 'query_id' in row and 'query' in row:
                queries[row['query_id'].strip()] = row['query']
    return queries

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Calculate overlap scores for retrieved documents.")

    # Inputs
    parser.add_argument("--vncorenlp", required=True, help="Path to VnCoreNLP folder")
    parser.add_argument("--stopwords", required=True, help="Path to stopwords TXT file")
    parser.add_argument("--doc_mapping", required=True, help="CSV containing doc_id and document")
    parser.add_argument("--query_mapping", required=True, help="CSV containing query_id and query")
    parser.add_argument("--input_jsonl", required=True, help="Input JSONL with 'topk_document_ids'")

    # Output
    parser.add_argument("--output_jsonl", required=True, help="Output path for overlap scores")

    args = parser.parse_args()

    # 1. Initialize Processor
    processor = VietnameseQueryProcessor(args.vncorenlp)
    calculator = LexicalCalculator(processor, args.stopwords)

    # 2. Load Data Mappings
    doc_map = load_document_map(args.doc_mapping)
    query_map = load_query_map(args.query_mapping)

    # Cache for tokenized text to avoid re-processing the same doc/query multiple times
    # (Though queries are likely unique per line, docs might be repeated across different queries)
    token_cache: Dict[str, Set[str]] = {}

    def get_tokens(text_id, text_content):
        if text_id not in token_cache:
            token_cache[text_id] = processor.process_text(text_content)
        return token_cache[text_id]

    print(f"Processing retrieval file: {args.input_jsonl}...")

    # Count lines for progress bar
    try:
        with open(args.input_jsonl, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for _ in f)
    except FileNotFoundError:
        print(f"Error: Input file {args.input_jsonl} not found.")
        sys.exit(1)

    with open(args.input_jsonl, 'r', encoding='utf-8') as fin, \
         open(args.output_jsonl, 'w', encoding='utf-8') as fout:

        for line in tqdm(fin, total=total_lines, desc="Calculating Overlaps"):
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            qid = str(entry.get('qid', '')).strip()
            topk_ids = entry.get('topk_document_ids', [])

            if not qid or qid not in query_map:
                continue

            # Process Query Tokens
            q_text = query_map[qid]
            q_tokens = get_tokens(f"q_{qid}", q_text)

            overlap_scores = {}

            for doc_id in topk_ids:
                doc_id = str(doc_id).strip()

                if doc_id not in doc_map:
                    # If doc text is missing, we can't calculate overlap
                    overlap_scores[doc_id] = 0.0
                    continue

                # Process Doc Tokens (Lazy load)
                d_text = doc_map[doc_id]
                d_tokens = get_tokens(f"d_{doc_id}", d_text)

                # Calculate Score
                score = calculator.calculate_overlap(q_tokens, d_tokens)
                overlap_scores[doc_id] = round(score, 4)

            # Construct Output Object
            output_entry = {
                "qid": qid,
                "document_overlap_scores": overlap_scores
            }

            fout.write(json.dumps(output_entry, ensure_ascii=False) + "\n")

    print(f"\nDone! Results saved to {args.output_jsonl}")

if __name__ == "__main__":
    main()
