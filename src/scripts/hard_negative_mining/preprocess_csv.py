import sys
import csv
import json
import string
import argparse
import os
import hashlib
from typing import Set
from tqdm import tqdm
import py_vncorenlp
from underthesea import text_normalize

# ==========================================
# CONFIGURATION
# ==========================================
STOPWORD_WHITELIST = {
    "không", "chẳng", "chả", "chưa", "phi", "vô", "tránh", "đừng", "chớ",
    "và", "hoặc", "nhưng", "tuy", "nếu", "thì", "vì", "do", "bởi", "tại", "nên", 
    "rằng", "là", "của", "thuộc",
    "tại", "ở", "trong", "ngoài", "trên", "dưới", "giữa", "với", "về", "đến",
    "ai", "gì", "nào", "đâu", "khi", "mấy", "bao_nhiêu", "thế_nào", "sao",
    "bị", "được", "do", "bởi"
}

class VietnameseProcessor:
    def __init__(self, vncorenlp_path: str, stopwords_path: str, use_whitelist: bool = False):
        # 1. Init VnCoreNLP
        if not os.path.exists(vncorenlp_path):
            raise FileNotFoundError(f"VnCoreNLP not found at {vncorenlp_path}")

        try:
            self.rdrsegmenter = py_vncorenlp.VnCoreNLP(
                save_dir=vncorenlp_path,
                annotators=["wseg"]
            )
        except Exception as e:
            print(f"Error init VnCoreNLP: {e}")
            sys.exit(1)

        # 2. Init Stopwords
        self.use_whitelist = use_whitelist
        self.stopwords = self._load_stopwords(stopwords_path)
        self.punctuation = set(string.punctuation)

    def _load_stopwords(self, path: str) -> Set[str]:
        sw = set()
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    w = line.strip().lower()
                    if w:
                        token = w.replace(' ', '_').replace('-', '_')
                        if self.use_whitelist and token in STOPWORD_WHITELIST:
                            continue
                        sw.add(token)
        return sw

    def process(self, text: str) -> str:
        """Returns a space-separated string of segmented tokens."""
        if not text: return ""
        try:
            text = text_normalize(text.lower())
        except:
            text = text.lower()

        try:
            # Segment
            sents = self.rdrsegmenter.word_segment(text)
            tokens = [t for sent in sents for t in sent.split()]
            # Filter
            valid = [t for t in tokens if t not in self.punctuation and t not in self.stopwords]
            return " ".join(valid)
        except:
            return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True, help="Path to CSV file containing Query and Document")
    parser.add_argument("--doc_mapping", required=True, help="CSV with columns: doc_id, document (Canonical Source)")
    parser.add_argument("--vncorenlp_path", required=True)
    parser.add_argument("--stopwords_path", required=True)
    parser.add_argument("--output_dir", required=True)

    # Column names
    parser.add_argument("--doc_col", default="document", help="Header name for the document column")
    parser.add_argument("--query_col", default="query", help="Header name for the query column")
    parser.add_argument("--map_doc_id", default="doc_id")
    parser.add_argument("--map_doc_col", default="document")
    parser.add_argument("--enable_whitelist", action="store_true", help="If set, prevents specific important stopwords (negations, logic) from being filtered out.")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    processor = VietnameseProcessor(args.vncorenlp_path, args.stopwords_path, use_whitelist=args.enable_whitelist)

    out_corpus_path = os.path.join(args.output_dir, "corpus_pretokenized.jsonl")
    out_query_path = os.path.join(args.output_dir, "queries_pretokenized.jsonl")
    
    # ---------------------------
    # 1. Process Document Mapping (Corpus & Lookup)
    # ---------------------------
    print(f">>> Processing Document Mapping: {args.doc_mapping}")

    # Map: Text -> Canonical ID (for resolving input_csv documents)
    canonical_text_to_id = {}

    csv.field_size_limit(sys.maxsize)

    with open(args.doc_mapping, 'r', encoding='utf-8') as f_in, \
         open(out_corpus_path, 'w', encoding='utf-8') as f_out:

        reader = csv.DictReader(f_in)
        for row in tqdm(reader, desc="Tokenizing Corpus"):
            doc_id = row.get(args.map_doc_id)
            content = row.get(args.map_doc_col, "").strip()

            if not doc_id or not content:
                continue

            # Tokenize Doc
            seg_doc = processor.process(content)

            # Write to Pyserini corpus
            obj = {"id": doc_id, "contents": seg_doc}
            f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")

            # Store mapping for lookup
            canonical_text_to_id[content] = doc_id

    # ---------------------------
    # 2. Process Input CSV (Queries)
    # ---------------------------
    print(f">>> Processing Input Queries: {args.input_csv}")

    missing_docs_count = 0
    resolved_count = 0

    with open(args.input_csv, 'r', encoding='utf-8') as f_in, \
         open(out_query_path, 'w', encoding='utf-8') as f_out:

        reader = csv.DictReader(f_in)
        for row in tqdm(reader, desc="Processing Queries"):
            doc_raw = row.get(args.doc_col, "").strip()
            query_raw = row.get(args.query_col, "").strip()

            if not doc_raw or not query_raw:
                continue

            # 1. Find Canonical ID
            canonical_id = canonical_text_to_id.get(doc_raw)

            if not canonical_id:
                missing_docs_count += 1
                continue

            # 2. Tokenize Query
            seg_query = processor.process(query_raw)
            if seg_query:
                # Write query linked to the canonical document ID
                query_obj = {
                    "pos_doc_id": canonical_id,
                    "queries": [{
                        "query_raw": query_raw,
                        "query_seg": seg_query
                    }]
                }
                f_out.write(json.dumps(query_obj, ensure_ascii=False) + "\n")
                resolved_count += 1

    print(f">>> Preprocessing complete.")
    print(f"    Files saved to {args.output_dir}")
    print(f"    Queries Processed: {resolved_count}")
    print(f"    Skipped (Doc text not found in mapping): {missing_docs_count}")

if __name__ == "__main__":
    main()
