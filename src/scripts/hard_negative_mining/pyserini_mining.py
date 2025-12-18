import sys
import csv
import json
import argparse
import os
import shutil
import subprocess
from tqdm import tqdm

# Pyserini Imports
try:
    from pyserini.search.lucene import LuceneSearcher
    from pyserini.pyclass import autoclass
except ImportError:
    print("Error: Pyserini not installed.", file=sys.stderr)
    sys.exit(1)

def load_doc_map(csv_path, id_col, doc_col):
    """Loads raw text to reconstruct the final triples."""
    m = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            m[row[id_col]] = row[doc_col]
    return m

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocessed_dir", required=True)
    parser.add_argument("--original_doc_csv", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--doc_col", default="document")
    parser.add_argument("--id_col", default="id")
    args = parser.parse_args()

    corpus_path = os.path.join(args.preprocessed_dir, "corpus_pretokenized.jsonl")
    queries_path = os.path.join(args.preprocessed_dir, "queries_pretokenized.jsonl")
    index_dir = os.path.join(args.preprocessed_dir, "pyserini_index")

    # ---------------------------------------------------------
    # STEP 1: INDEXING (Subprocess)
    # ---------------------------------------------------------
    if os.path.exists(index_dir):
        shutil.rmtree(index_dir)

    print(">>> Building Pyserini Index...")
    # We must point Pyserini to the FOLDER containing the jsonl, not the file itself
    # So we create a subdir for the corpus file
    temp_corpus_dir = os.path.join(args.preprocessed_dir, "corpus_folder")
    os.makedirs(temp_corpus_dir, exist_ok=True)
    shutil.copy(corpus_path, os.path.join(temp_corpus_dir, "docs.jsonl"))

    cmd = [
        "python", "-m", "pyserini.index.lucene",
        "--collection", "JsonCollection",
        "--input", temp_corpus_dir,
        "--index", index_dir,
        "--generator", "DefaultLuceneDocumentGenerator",
        "--threads", "1",
        "--storePositions", "--storeDocvectors", "--storeRaw",
        "--pretokenized"  # CRITICAL FLAG
    ]
    subprocess.check_call(cmd)

    # Cleanup corpus folder
    shutil.rmtree(temp_corpus_dir)

    # ---------------------------------------------------------
    # STEP 2: SEARCHING
    # ---------------------------------------------------------
    print(">>> Loading Raw Documents for reconstruction...")
    doc_map = load_doc_map(args.original_doc_csv, args.id_col, args.doc_col)

    print(">>> Initializing Searcher...")
    searcher = LuceneSearcher(index_dir)
    searcher.set_bm25(k1=1.2, b=0.75)

    # Force WhitespaceAnalyzer to respect our segmentation
    try:
        JWhitespaceAnalyzer = autoclass('org.apache.lucene.analysis.core.WhitespaceAnalyzer')
        searcher.set_analyzer(JWhitespaceAnalyzer())
    except Exception as e:
        print(f"Warning: Could not set WhitespaceAnalyzer: {e}")

    print(">>> Mining Negatives...")
    with open(queries_path, 'r', encoding='utf-8') as f_in, \
         open(args.output_jsonl, 'w', encoding='utf-8') as f_out:

        for line in tqdm(f_in):
            data = json.loads(line)
            pos_id = data['pos_doc_id']

            if pos_id not in doc_map:
                continue

            for q_data in data['queries']:
                raw_query = q_data['query_raw']
                seg_query = q_data['query_seg']

                # Search using the SEGMENTED string
                hits = searcher.search(seg_query, k=args.top_k + 10)

                hard_negatives = []
                for hit in hits:
                    if hit.docid == pos_id:
                        continue

                    # Retrieve raw text from our map
                    raw_neg = doc_map.get(hit.docid)
                    if raw_neg:
                        hard_negatives.append(raw_neg)

                    if len(hard_negatives) >= args.top_k:
                        break

                # Write Output: Query(Raw), Pos(Raw), Neg(Raw)
                out = {
                    "query": raw_query,
                    "pos": [doc_map[pos_id]],
                    "neg": hard_negatives
                }
                f_out.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f">>> Done. Output at {args.output_jsonl}")

if __name__ == "__main__":
    main()
