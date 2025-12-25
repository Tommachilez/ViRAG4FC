import sys
import csv
import json
import string
import argparse
import os
from typing import Set, Dict
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


def load_csv_to_map(csv_path: str, id_col: str, text_col: str) -> Dict[str, str]:
    """Reads a CSV and returns a mapping of ID -> Text."""
    mapping = {}
    csv.field_size_limit(sys.maxsize)

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = row.get(id_col)
            text = row.get(text_col, "").strip()
            if rid and text:
                mapping[rid] = text
    return mapping


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full_data_csv", required=True, help="Original full dataset (for generated query ID lookup)")
    parser.add_argument("--doc_mapping", required=True, help="CSV with columns: doc_id, document")
    # parser.add_argument("--query_mapping", required=True, help="CSV with columns: query_id, query")
    parser.add_argument("--query_jsonl", required=True, help="Generated queries JSONL")
    parser.add_argument("--train_csv", required=False, help="Training data CSV with schema {document, evidence, claim, id}")

    parser.add_argument("--vncorenlp_path", required=True)
    parser.add_argument("--stopwords_path", required=True)
    parser.add_argument("--output_dir", required=True)

    # Column configuration
    parser.add_argument("--doc_col", default="document")
    parser.add_argument("--id_col", default="id")
    parser.add_argument("--claim_col", default="query", help="Column name for claim in train_csv")
    parser.add_argument("--map_doc_id", default="doc_id")
    parser.add_argument("--map_doc_col", default="document")

    parser.add_argument("--enable_whitelist", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    processor = VietnameseProcessor(args.vncorenlp_path, args.stopwords_path, use_whitelist=args.enable_whitelist)

    print(">>> Loading Document Mapping (Canonical IDs)...")
    canonical_text_to_id = {}

    # Check if doc_mapping exists
    if not os.path.exists(args.doc_mapping):
        raise FileNotFoundError(f"Doc mapping not found at {args.doc_mapping}")

    # ---------------------------------------------------------
    # PART 1: Process Documents (CSV -> JSONL for Pyserini)
    # ---------------------------------------------------------
    print(">>> Processing Documents from Mapping...")
    out_corpus_path = os.path.join(args.output_dir, "corpus_pretokenized.jsonl")

    with open(args.doc_mapping, 'r', encoding='utf-8') as f_in, \
         open(out_corpus_path, 'w', encoding='utf-8') as f_out:

        reader = csv.DictReader(f_in)
        for row in tqdm(reader, desc="Tokenizing Corpus"):
            doc_id = row.get(args.map_doc_id)
            content = row.get(args.map_doc_col, "").strip()

            if not doc_id or not content:
                continue

            # Tokenize
            seg_text = processor.process(content)

            # Write to Pyserini corpus
            obj = {"id": doc_id, "contents": seg_text}
            f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")

            # Store for query resolution
            canonical_text_to_id[content] = doc_id

    # ---------------------------------------------------------
    # PART 2: Process Queries (JSONL -> JSONL with seg)
    # ---------------------------------------------------------

    print(">>> Loading Full Data CSV (Original IDs)...")
    # We need Original ID -> Text to bridge the gap
    original_id_to_text = load_csv_to_map(args.full_data_csv, args.id_col, args.doc_col)

    print(">>> Processing Queries & Resolving IDs...")
    out_query_path = os.path.join(args.output_dir, "queries_pretokenized.jsonl")

    missing_docs_count = 0
    resolved_count = 0

    with open(out_query_path, 'w', encoding='utf-8') as f_out:

        # --- 3a. Process Generated Queries (JSONL) ---
        print(f"   > Processing Generated Queries from {args.query_jsonl}")
        with open(args.query_jsonl, 'r', encoding='utf-8') as f_in:
            for line in tqdm(f_in, desc="Generated Queries"):
                try:
                    data = json.loads(line)
                    original_pos_id = data.get(args.id_col)

                    # 1. Get raw text from original full data
                    raw_doc_text = original_id_to_text.get(original_pos_id)
                    if not raw_doc_text:
                        continue

                    # 2. Find Canonical ID using the text
                    canonical_id = canonical_text_to_id.get(raw_doc_text)
                    if not canonical_id:
                        missing_docs_count += 1
                        continue

                    # 3. Process generated queries
                    processed_queries = []
                    for q_obj in data.get("generated_queries", []):
                        raw_q = q_obj.get("query", "")
                        seg_q = processor.process(raw_q)
                        if seg_q:
                            processed_queries.append({
                                "query_raw": raw_q,
                                "query_seg": seg_q
                            })

                    if processed_queries:
                        out_obj = {
                            "pos_doc_id": canonical_id, 
                            "queries": processed_queries
                        }
                        f_out.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
                        resolved_count += 1

                except json.JSONDecodeError:
                    continue

        # --- 3b. Process Real Claims (Train CSV) ---
        if args.train_csv and os.path.exists(args.train_csv):
            print(f"   > Processing Claims from {args.train_csv}")
            csv.field_size_limit(sys.maxsize)

            with open(args.train_csv, 'r', encoding='utf-8') as f_train:
                reader = csv.DictReader(f_train)

                for row in tqdm(reader, desc="Train Claims"):
                    claim = row.get(args.claim_col)
                    doc_text = row.get(args.doc_col) # Uses same document col name as Full Data

                    if not claim or not doc_text:
                        continue

                    doc_text = doc_text.strip()

                    # 1. Find Canonical ID directly using the text (Train CSV has text)
                    canonical_id = canonical_text_to_id.get(doc_text)

                    if not canonical_id:
                        # If the doc text in train_csv isn't in doc_mapping, we can't build a pair
                        missing_docs_count += 1
                        continue

                    # 2. Tokenize the Claim
                    seg_claim = processor.process(claim)
                    if not seg_claim:
                        continue

                    # 3. Write Output
                    out_obj = {
                        "pos_doc_id": canonical_id,
                        "queries": [{
                            "query_raw": claim,
                            "query_seg": seg_claim
                        }]
                    }
                    f_out.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
                    resolved_count += 1
        else:
            if args.train_csv:
                print(f"Warning: train_csv path provided but file does not exist: {args.train_csv}")

    print(">>> Preprocessing complete.")
    print(f"    Files saved to {args.output_dir}")
    print(f"    Queries Resolved: {resolved_count}")
    print(f"    Queries Dropped (Text mismatch between Full Data and Mapping): {missing_docs_count}")

if __name__ == "__main__":
    main()
