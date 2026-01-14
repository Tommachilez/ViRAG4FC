import sys
import csv
import argparse
import os
import shutil
import subprocess
import collections
import numpy as np
from tqdm import tqdm

# Pyserini Imports
try:
    from pyserini.search.lucene import LuceneSearcher
    from pyserini.pyclass import autoclass
except ImportError:
    print("Error: Pyserini not installed.", file=sys.stderr)
    sys.exit(1)

def build_index(preprocessed_dir, index_dir, threads=2):
    """Builds Pyserini index from corpus_pretokenized.jsonl if it doesn't exist."""
    corpus_path = os.path.join(preprocessed_dir, "corpus_pretokenized.jsonl")

    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

    print(f">>> Building Index at {index_dir}...")
    temp_corpus_dir = os.path.join(preprocessed_dir, "temp_index_corpus")
    os.makedirs(temp_corpus_dir, exist_ok=True)

    # Structure for Pyserini (folder/file.jsonl)
    dest_corpus = os.path.join(temp_corpus_dir, "docs.jsonl")
    if not os.path.exists(dest_corpus):
        shutil.copy(corpus_path, dest_corpus)

    cmd = [
        "python", "-m", "pyserini.index.lucene",
        "--collection", "JsonCollection",
        "--input", temp_corpus_dir,
        "--index", index_dir,
        "--generator", "DefaultLuceneDocumentGenerator",
        "--threads", str(threads),
        "--storePositions", "--storeDocvectors", "--storeRaw",
        "--pretokenized"
    ]
    subprocess.check_call(cmd)
    shutil.rmtree(temp_corpus_dir)
    print(">>> Indexing Complete.")

def load_segmented_queries(query_path):
    """Loads ALREADY SEGMENTED queries from TSV."""
    queries = {}
    print(">>> Loading Segmented Queries...")
    with open(query_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if len(row) >= 2:
                q_id, q_text = row[0], row[1]
                queries[q_id] = q_text.strip()
    return queries

def load_qrels(qrels_path):
    qrels = collections.defaultdict(dict)
    with open(qrels_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 0: continue

            # Support 4-col (TREC) or 3-col formats
            if len(parts) == 4:
                q_id, _, doc_id, rel = parts
            elif len(parts) >= 3:
                q_id, doc_id, rel = parts[0], parts[1], parts[2]
            else:
                continue

            if int(rel) > 0:
                qrels[q_id][doc_id] = int(rel)
    return qrels

def calculate_metrics(retrieved_docs, ground_truth, k_values=[10, 50, 100]):
    metrics = {}
    relevant_retrieved = [1 if doc_id in ground_truth else 0 for doc_id in retrieved_docs]

    if not ground_truth:
        return {f"R@{k}": 0.0 for k in k_values}, 0.0

    total_relevant = len(ground_truth)
    for k in k_values:
        hits = sum(relevant_retrieved[:k])
        metrics[f"R@{k}"] = hits / total_relevant if total_relevant > 0 else 0.0

    mrr = 0.0
    for i, is_rel in enumerate(relevant_retrieved[:10]):
        if is_rel:
            mrr = 1.0 / (i + 1)
            break

    return metrics, mrr

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocessed_dir", required=True, help="Dir with corpus_pretokenized.jsonl")
    parser.add_argument("--queries_tsv", required=True, help="Path to SEGMENTED test_queries.tsv")
    parser.add_argument("--qrels_tsv", required=True, help="Path to test_qrels.tsv")
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()

    # 1. Setup Index
    index_dir = os.path.join(args.preprocessed_dir, "pyserini_index")
    if not os.path.exists(index_dir) or not os.listdir(index_dir):
        build_index(args.preprocessed_dir, index_dir, args.threads)
    else:
        print(f">>> Using existing index at {index_dir}")

    # 2. Init Searcher
    print(">>> Initializing LuceneSearcher...")
    searcher = LuceneSearcher(index_dir)
    searcher.set_bm25(k1=1.2, b=0.75)

    try:
        JWhitespaceAnalyzer = autoclass('org.apache.lucene.analysis.core.WhitespaceAnalyzer')
        searcher.set_analyzer(JWhitespaceAnalyzer())
        print(">>> WhitespaceAnalyzer set.")
    except Exception as e:
        print(f"Warning: Could not set WhitespaceAnalyzer: {e}")

    # 3. Load Data
    queries = load_segmented_queries(args.queries_tsv)
    qrels = load_qrels(args.qrels_tsv)

    # 4. Search & Eval
    print(f">>> Searching (top_k={args.top_k})...")

    cumulative_metrics = collections.defaultdict(list)
    k_list = [3, 10, 50, args.top_k]

    for q_id, q_text in tqdm(queries.items()):
        if q_id not in qrels:
            continue

        try:
            hits = searcher.search(q_text, k=args.top_k)
        except Exception:
            hits = []

        retrieved_ids = [hit.docid for hit in hits]
        ground_truth = qrels[q_id]

        met, mrr = calculate_metrics(retrieved_ids, ground_truth, k_values=k_list)

        for k_val in k_list:
            cumulative_metrics[f"R@{k_val}"].append(met[f"R@{k_val}"])
        cumulative_metrics["MRR@10"].append(mrr)

    # 5. Results
    print("\n" + "="*30)
    print("       EVALUATION RESULTS       ")
    print("="*30)

    if len(cumulative_metrics["MRR@10"]) > 0:
        print(f"Queries Evaluated: {len(cumulative_metrics['MRR@10'])}")
        print(f"MRR@10:    {np.mean(cumulative_metrics['MRR@10']):.4f}")
        for k_val in k_list:
            print(f"Recall@{k_val}: {np.mean(cumulative_metrics[f'R@{k_val}']):.4f}")
    else:
        print("No matching query IDs found.")

if __name__ == "__main__":
    main()
