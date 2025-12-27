import sys
import csv
import argparse
import os
import shutil
import subprocess
import collections
import string
import numpy as np
from tqdm import tqdm
from typing import Set

# NLP Imports (Must match preprocessing environment)
try:
    import py_vncorenlp
    from underthesea import text_normalize
except ImportError:
    print("Error: NLP libraries missing. Run: pip install py_vncorenlp underthesea", file=sys.stderr)
    sys.exit(1)

# Pyserini Imports
try:
    from pyserini.search.lucene import LuceneSearcher
    from pyserini.pyclass import autoclass
except ImportError:
    print("Error: Pyserini not installed.", file=sys.stderr)
    sys.exit(1)

# ==========================================
# 1. TEXT PROCESSOR (Copied from preprocess_csv.py)
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
            # Suppress print statements from vncorenlp if possible, or just init
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

# ==========================================
# 2. INDEXING & RETRIEVAL LOGIC
# ==========================================

def build_index(preprocessed_dir, index_dir, threads=2):
    """Builds Pyserini index if it doesn't exist."""
    corpus_path = os.path.join(preprocessed_dir, "corpus_pretokenized.jsonl")

    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

    print(f">>> Building Index at {index_dir}...")
    temp_corpus_dir = os.path.join(preprocessed_dir, "temp_index_corpus")
    os.makedirs(temp_corpus_dir, exist_ok=True)

    # Copy file to a structure Pyserini likes (folder/file.jsonl)
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

def load_tsv_queries(query_path, processor):
    """
    Loads raw queries from TSV and segments them.
    Format: query_id \t raw_query_text
    """
    queries = {}
    print(">>> Loading and Segmenting Queries...")
    with open(query_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in tqdm(reader, desc="Segmenting"):
            if len(row) >= 2:
                q_id, q_raw = row[0], row[1]
                # APPLY SEGMENTATION HERE
                q_seg = processor.process(q_raw)
                if q_seg:
                    queries[q_id] = q_seg
    return queries

def load_qrels(qrels_path):
    """
    Loads qrels. 
    Supports 3-col (qid docid rel) or 4-col (qid 0 docid rel)
    """
    qrels = collections.defaultdict(dict)
    with open(qrels_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 0: continue

            if len(parts) == 4:
                q_id, _, doc_id, rel = parts
            elif len(parts) >= 2:
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
    parser.add_argument("--queries_tsv", required=True, help="Path to test_queries.tsv (Raw Text)")
    parser.add_argument("--qrels_tsv", required=True, help="Path to test_qrels.tsv")
    parser.add_argument("--vncorenlp_path", required=True, help="Path to VnCoreNLP models")
    parser.add_argument("--stopwords_path", required=True, help="Path to stopwords-vi.txt")
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()

    # 1. Initialize Processor (Same as preprocess_csv.py)
    print(">>> Initializing NLP Processor...")
    processor = VietnameseProcessor(args.vncorenlp_path, args.stopwords_path, use_whitelist=True)

    # 2. Setup Index
    index_dir = os.path.join(args.preprocessed_dir, "pyserini_index")
    if not os.path.exists(index_dir) or not os.listdir(index_dir):
        build_index(args.preprocessed_dir, index_dir, args.threads)
    else:
        print(f">>> Using existing index at {index_dir}")

    # 3. Initialize Searcher
    print(">>> Initializing LuceneSearcher...")
    searcher = LuceneSearcher(index_dir)
    searcher.set_bm25(k1=1.2, b=0.75)

    # Use WhitespaceAnalyzer for pre-segmented text
    try:
        JWhitespaceAnalyzer = autoclass('org.apache.lucene.analysis.core.WhitespaceAnalyzer')
        searcher.set_analyzer(JWhitespaceAnalyzer())
    except Exception as e:
        print(f"Warning: Could not set WhitespaceAnalyzer: {e}")

    # 4. Load & Segment Queries
    queries = load_tsv_queries(args.queries_tsv, processor)
    qrels = load_qrels(args.qrels_tsv)

    print(f"    Loaded {len(queries)} valid queries.")
    print(f"    Loaded {len(qrels)} qrels.")

    # 5. Search & Eval
    print(f">>> Searching (top_k={args.top_k})...")

    cumulative_metrics = collections.defaultdict(list)
    k_list = [10, 50, args.top_k]

    for q_id, q_seg_text in tqdm(queries.items(), desc="Retrieving"):
        if q_id not in qrels:
            continue

        try:
            hits = searcher.search(q_seg_text, k=args.top_k)
        except Exception as e:
            # Handle empty queries or search errors
            hits = []

        retrieved_ids = [hit.docid for hit in hits]
        ground_truth = qrels[q_id]

        met, mrr = calculate_metrics(retrieved_ids, ground_truth, k_values=k_list)

        for k_val in k_list:
            cumulative_metrics[f"R@{k_val}"].append(met[f"R@{k_val}"])
        cumulative_metrics["MRR@10"].append(mrr)

    # 6. Results
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
