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

def load_csv_pos_map(csv_path, id_col, doc_col):
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
    map_path = os.path.join(args.preprocessed_dir, "dedup_docs_map.json")
    index_dir = os.path.join(args.preprocessed_dir, "pyserini_index")

    print(">>> Loading Deduplicated Document Map...")
    with open(map_path, 'r', encoding='utf-8') as f:
        # Maps Hash (Index ID) -> Raw Text
        hash_to_text = json.load(f)

    print(">>> Loading Original CSV for Positive Doc lookup...")
    # Maps Row ID (Query Source) -> Raw Text
    row_id_to_text = load_csv_pos_map(args.original_doc_csv, args.id_col, args.doc_col)

    # ---------------------------------------------------------
    # STEP 1: INDEXING (Subprocess)
    # ---------------------------------------------------------
    if os.path.exists(index_dir):
        shutil.rmtree(index_dir)

    print(">>> Building Pyserini Index...")
    temp_corpus_dir = os.path.join(args.preprocessed_dir, "corpus_folder")
    os.makedirs(temp_corpus_dir, exist_ok=True)
    shutil.copy(corpus_path, os.path.join(temp_corpus_dir, "docs.jsonl"))

    cmd = [
        "python", "-m", "pyserini.index.lucene",
        "--collection", "JsonCollection",
        "--input", temp_corpus_dir,
        "--index", index_dir,
        "--generator", "DefaultLuceneDocumentGenerator",
        "--threads", "2",
        "--storePositions", "--storeDocvectors", "--storeRaw",
        "--pretokenized"
    ]
    subprocess.check_call(cmd)
    shutil.rmtree(temp_corpus_dir)

    # ---------------------------------------------------------
    # STEP 2: SEARCHING
    # ---------------------------------------------------------
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

            pos_text_raw = row_id_to_text.get(pos_id)
            if not pos_text_raw:
                continue

            for q_data in data['queries']:
                raw_query = q_data['query_raw']
                seg_query = q_data['query_seg']

                # Search using the SEGMENTED string
                hits = searcher.search(seg_query, k=args.top_k + 10)

                candidates = {}
                # Add Positive
                candidates[pos_id] = pos_text_raw

                for hit in hits:
                    candidate_text = hash_to_text.get(hit.docid)

                    if not candidate_text:
                        continue

                    if candidate_text == pos_text_raw:
                        continue

                    candidates[hit.docid] = candidate_text

                    if len(candidates) >= args.top_k + 1:  # +1 for positive
                        break

                # Write Output
                out = { "query": raw_query, "candidates": candidates }
                f_out.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f">>> Done. Output at {args.output_jsonl}")

if __name__ == "__main__":
    main()
