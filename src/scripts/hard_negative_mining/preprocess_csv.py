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

def get_md5(text: str) -> str:
    """Generates a consistent hash for deduplication."""
    return hashlib.md5(text.strip().encode('utf-8')).hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_tsv", required=True, help="Path to TSV file containing Query and Document")
    parser.add_argument("--vncorenlp_path", required=True)
    parser.add_argument("--stopwords_path", required=True)
    parser.add_argument("--output_dir", required=True)

    # Column names in the TSV
    parser.add_argument("--doc_col", default="document", help="Header name for the document column")
    parser.add_argument("--query_col", default="query", help="Header name for the query column")
    parser.add_argument("--enable_whitelist", action="store_true", help="If set, prevents specific important stopwords (negations, logic) from being filtered out.")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    processor = VietnameseProcessor(args.vncorenlp_path, args.stopwords_path, use_whitelist=args.enable_whitelist)

    out_corpus_path = os.path.join(args.output_dir, "corpus_pretokenized.jsonl")
    out_query_path = os.path.join(args.output_dir, "queries_pretokenized.jsonl")
    out_map_path = os.path.join(args.output_dir, "dedup_docs_map.json")

    # Deduplication tracking
    seen_hashes = set()
    doc_hash_to_raw = {}

    print(f">>> Processing TSV: {args.input_tsv}")

    # Open all files at once
    with open(args.input_tsv, 'r', encoding='utf-8') as f_in, \
         open(out_corpus_path, 'w', encoding='utf-8') as f_corpus, \
         open(out_query_path, 'w', encoding='utf-8') as f_queries:

        # Use DictReader (assumes CSV has headers)
        reader = csv.DictReader(f_in)

        for row in tqdm(reader, desc="Processing rows"):
            doc_raw = row.get(args.doc_col, "").strip()
            query_raw = row.get(args.query_col, "").strip()

            if not doc_raw:
                continue

            # ---------------------------
            # 1. Process Document
            # ---------------------------
            doc_hash = get_md5(doc_raw)

            # If we haven't seen this doc text before, segment it and add to corpus
            if doc_hash not in seen_hashes:
                seen_hashes.add(doc_hash)

                # Tokenize Doc
                seg_doc = processor.process(doc_raw)

                # Write to Corpus (ID = Hash)
                obj = {"id": doc_hash, "contents": seg_doc}
                f_corpus.write(json.dumps(obj, ensure_ascii=False) + "\n")

                # Keep map for mining step
                doc_hash_to_raw[doc_hash] = doc_raw

            # ---------------------------
            # 2. Process Query
            # ---------------------------
            if query_raw:
                seg_query = processor.process(query_raw)
                if seg_query:
                    # Write query with link to the positive document Hash
                    query_obj = {
                        "pos_doc_id": doc_hash,
                        "queries": [{
                            "query_raw": query_raw,
                            "query_seg": seg_query
                        }]
                    }
                    f_queries.write(json.dumps(query_obj, ensure_ascii=False) + "\n")

    print(f">>> Saving document map ({len(doc_hash_to_raw)} unique docs)...")
    with open(out_map_path, 'w', encoding='utf-8') as f:
        json.dump(doc_hash_to_raw, f, ensure_ascii=False)

    print(f">>> Preprocessing complete. Files saved to {args.output_dir}")

if __name__ == "__main__":
    main()
