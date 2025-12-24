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

def sanitize_text(text):
    """Removes tabs and newlines for safe TSV writing."""
    if not text:
        return ""
    return text.replace('\t', ' ').replace('\n', ' ').strip()

def load_csv_maps(csv_path, id_col, doc_col):
    """
    Loads raw text to reconstruct final triples.
    Returns:
        id_to_text: Map of Original ID -> Text
        text_to_id: Map of Text -> Original ID (for reverse lookup)
    """
    id_to_text = {}
    text_to_id = {}

    # Increase field size limit for large CSV fields
    csv.field_size_limit(sys.maxsize)

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = row[id_col]
            txt = row[doc_col]

            id_to_text[rid] = txt

            # If multiple IDs have the exact same text, this will keep the first one found.
            # This is acceptable since they are duplicates.
            if txt not in text_to_id:
                text_to_id[txt] = rid

    return id_to_text, text_to_id

def save_unique_mapping(text_to_id_map, output_file_path):
    """
    Saves the mapping of Canonical ID -> Unique Text to a TSV file.
    This ensures downstream scripts know exactly which text corresponds to the IDs in the JSONL.
    """
    output_dir = os.path.dirname(output_file_path)
    # If output_dir is empty (current dir), leave it as is, otherwise join
    mapping_path = os.path.join(output_dir, "document_unique_mapping.tsv") if output_dir else "document_unique_mapping.tsv"

    print(f">>> Saving unique document mapping to {mapping_path}...")

    try:
        with open(mapping_path, 'w', encoding='utf-8') as f:
            for text, rid in text_to_id_map.items():
                clean_text = sanitize_text(text)
                f.write(f"{rid}\t{clean_text}\n")
        print(f">>> Mapping saved. ({len(text_to_id_map)} unique documents)")
    except Exception as e:
        print(f"Error saving mapping file: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocessed_dir", required=True)
    parser.add_argument("--original_doc_csv", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--top_k", type=int, default=50)
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

    print(">>> Loading Original CSV for ID lookup...")
    # row_id_to_text: Used to get the query's positive doc text
    # text_to_row_id: Used to convert retrieved candidates back to original IDs
    row_id_to_text, text_to_row_id = load_csv_maps(args.original_doc_csv, args.id_col, args.doc_col)

    save_unique_mapping(text_to_row_id, args.output_jsonl)

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
                # Add Positive (Using Original ID)
                candidates[pos_id] = pos_text_raw

                for hit in hits:
                    # hit.docid is the HASH
                    candidate_text = hash_to_text.get(hit.docid)

                    if not candidate_text:
                        continue

                    # Filter out exact positive text match (deduplication)
                    if candidate_text == pos_text_raw:
                        continue

                    # RETRIEVE ORIGINAL ID
                    # We use the text to find the original CSV ID
                    original_doc_id = text_to_row_id.get(candidate_text)

                    # Fallback: if somehow not found (unlikely), use the hash
                    final_id = original_doc_id if original_doc_id else hit.docid

                    candidates[final_id] = candidate_text

                    if len(candidates) >= args.top_k + 1:  # +1 for positive
                        break

                # Write Output
                out = { "query": raw_query, "candidates": candidates }
                f_out.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f">>> Done. Output at {args.output_jsonl}")

if __name__ == "__main__":
    main()
