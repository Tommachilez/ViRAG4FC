import sys
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

def save_unique_mapping(hash_to_text_map, output_file_path):
    """
    Saves the mapping of Hash ID -> Unique Text to a TSV file.
    """
    output_dir = os.path.dirname(output_file_path)
    mapping_path = os.path.join(output_dir, "document_unique_mapping.tsv") if output_dir else "document_unique_mapping.tsv"

    print(f">>> Saving unique document mapping to {mapping_path}...")
    try:
        with open(mapping_path, 'w', encoding='utf-8') as f:
            f.write("id\tdocument\n")
            for doc_id, text in hash_to_text_map.items():
                clean_text = sanitize_text(text)
                f.write(f"{doc_id}\t{clean_text}\n")
        print(f">>> Mapping saved. ({len(hash_to_text_map)} unique documents)")
    except Exception as e:
        print(f"Error saving mapping file: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocessed_dir", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--top_k", type=int, default=50)
    args = parser.parse_args()

    # Paths setup
    corpus_path = os.path.join(args.preprocessed_dir, "corpus_pretokenized.jsonl")
    queries_path = os.path.join(args.preprocessed_dir, "queries_pretokenized.jsonl")
    map_path = os.path.join(args.preprocessed_dir, "dedup_docs_map.json")
    index_dir = os.path.join(args.preprocessed_dir, "pyserini_index")

    # 1. Load Map
    print(">>> Loading Deduplicated Document Map...")
    if not os.path.exists(map_path):
        print(f"Error: Map file not found at {map_path}")
        sys.exit(1)
        
    with open(map_path, 'r', encoding='utf-8') as f:
        # Maps Hash (Index ID) -> Raw Text
        hash_to_text = json.load(f)

    # Save mapping for user reference (Hash -> Text)
    save_unique_mapping(hash_to_text, args.output_jsonl)

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

    # Force WhitespaceAnalyzer
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
            # pos_id is now the HASH from preprocess
            pos_id = data['pos_doc_id'] 

            pos_text_raw = hash_to_text.get(pos_id)
            if not pos_text_raw:
                continue

            for q_data in data['queries']:
                raw_query = q_data['query_raw']
                seg_query = q_data['query_seg']

                # Search
                hits = searcher.search(seg_query, k=args.top_k + 10)

                candidates = {}
                # Add Positive
                candidates[pos_id] = pos_text_raw

                for hit in hits:
                    # hit.docid is the Hash in the index
                    candidate_hash = hit.docid
                    candidate_text = hash_to_text.get(candidate_hash)

                    if not candidate_text:
                        continue

                    # Filter out exact positive text or exact ID match
                    if candidate_hash == pos_id or candidate_text == pos_text_raw:
                        continue

                    candidates[candidate_hash] = candidate_text

                    if len(candidates) >= args.top_k + 1:
                        break

                # Write Output
                out = { "query": raw_query, "candidates": candidates }
                f_out.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f">>> Done. Output at {args.output_jsonl}")

if __name__ == "__main__":
    main()
