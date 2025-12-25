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
        self.stopwords = self._load_stopwords(stopwords_path)
        self.punctuation = set(string.punctuation)
        self.use_whitelist = use_whitelist

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
    parser.add_argument("--doc_csv", required=True)
    parser.add_argument("--query_jsonl", required=True)
    parser.add_argument("--vncorenlp_path", required=True)
    parser.add_argument("--stopwords_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--doc_col", default="document")
    parser.add_argument("--id_col", default="id")
    parser.add_argument("--enable_whitelist", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    processor = VietnameseProcessor(args.vncorenlp_path, args.stopwords_path, use_whitelist=args.enable_whitelist)

    # ---------------------------------------------------------
    # PART 1: Process Documents (CSV -> JSONL for Pyserini)
    # ---------------------------------------------------------
    print(">>> Processing Documents (Deduplicating)...")
    out_corpus_path = os.path.join(args.output_dir, "corpus_pretokenized.jsonl")
    out_map_path = os.path.join(args.output_dir, "dedup_docs_map.json")

    # Tracking sets for deduplication
    seen_hashes = set()
    doc_hash_to_raw = {}

    with open(args.doc_csv, 'r', encoding='utf-8') as f_in, \
         open(out_corpus_path, 'w', encoding='utf-8') as f_out:

        reader = csv.DictReader(f_in)
        for row in tqdm(reader, desc="Deduping Docs"):
            content = row.get(args.doc_col, "").strip()

            if not content:
                continue

            doc_hash = get_md5(content)
            if doc_hash not in seen_hashes:
                seen_hashes.add(doc_hash)

                # Tokenize
                seg_text = processor.process(content)

                # Write to Pyserini corpus (ID = Hash)
                obj = {"id": doc_hash, "contents": seg_text}
                f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")

                # Store mapping for later retrieval
                doc_hash_to_raw[doc_hash] = content

    # Save the map so pyserini_mining.py can look up raw text
    print(f">>> Saving deduplicated document map ({len(doc_hash_to_raw)} unique docs)...")
    with open(out_map_path, 'w', encoding='utf-8') as f:
        json.dump(doc_hash_to_raw, f, ensure_ascii=False)

    # ---------------------------------------------------------
    # PART 2: Process Queries (JSONL -> JSONL with seg)
    # ---------------------------------------------------------
    print(">>> Processing Queries...")
    out_query_path = os.path.join(args.output_dir, "queries_pretokenized.jsonl")

    with open(args.query_jsonl, 'r', encoding='utf-8') as f_in, \
         open(out_query_path, 'w', encoding='utf-8') as f_out:

        for line in tqdm(f_in, desc="Queries"):
            try:
                data = json.loads(line)
                # We need to process every generated query
                processed_queries = []
                for q_obj in data.get("generated_queries", []):
                    raw_q = q_obj.get("query", "")
                    seg_q = processor.process(raw_q)
                    if seg_q:
                        processed_queries.append({
                            "query_raw": raw_q,
                            "query_seg": seg_q
                        })

                # Save a structure that the Mining script can easily use
                if processed_queries:
                    out_obj = {
                        "pos_doc_id": data.get(args.id_col),
                        "queries": processed_queries
                    }
                    f_out.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            except json.JSONDecodeError:
                continue

    print(f">>> Preprocessing complete. Files saved to {args.output_dir}")

if __name__ == "__main__":
    main()
