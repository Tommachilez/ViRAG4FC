import sys
import csv
import argparse
import os
import json
import ast
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report

# Pyserini Imports (JVM Context)
try:
    from pyserini.search.lucene import LuceneSearcher
    from pyserini.pyclass import autoclass
except ImportError:
    print("Error: Pyserini not installed.", file=sys.stderr)
    sys.exit(1)

# Local Import
from src.scripts.reader_llm import ReaderLLM

# ==========================================
# 1. DATA LOADING HELPERS
# ==========================================

def load_doc_mapping(csv_path):
    """Loads Canonical ID -> Raw Text for the LLM."""
    print(f"Loading document text from {csv_path}...")
    doc_map = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc="Loading Docs"):
            rid = row.get('doc_id', '').strip()
            txt = row.get('document', '').strip()
            if rid and txt:
                doc_map[rid] = txt
    return doc_map

def load_segmented_queries(tsv_path):
    """Loads the pre-segmented queries (output of segment_queries.py)."""
    print(f"Loading segmented queries from {tsv_path}...")
    queries = {}
    with open(tsv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if len(row) >= 2:
                q_id = row[0].strip()
                q_seg = row[1].strip()
                queries[q_id] = q_seg
    return queries

def load_raw_queries(csv_path):
    """Loads raw natural language queries (for the LLM Prompt)."""
    print(f"Loading raw queries from {csv_path}...")
    queries = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            queries[row['query_id'].strip()] = row['query'].strip()
    return queries

def load_labels(label_file_path, raw_queries_map):
    """Loads ground truth verdicts (Supported/Refuted) for Evaluation."""
    print(f"Loading labels from {label_file_path}...")
    try:
        df = pd.read_csv(label_file_path)
    except Exception as e:
        print(f"Error reading label file: {e}")
        return {}

    # Map Query Text -> Label
    query_text_to_label = dict(zip(df['query'].str.strip(), df['label'].str.strip()))

    qid_to_label = {}
    for qid, q_text in raw_queries_map.items():
        clean_text = q_text.strip()
        if clean_text in query_text_to_label:
            qid_to_label[qid] = query_text_to_label[clean_text]
    return qid_to_label

def parse_llm_output(output_str):
    if isinstance(output_str, dict): return output_str
    try: return json.loads(output_str)
    except: pass
    try: return ast.literal_eval(output_str)
    except: pass
    return {"qid": None, "verdict": "Error", "explanation": f"Parse Error: {str(output_str)[:100]}"}

def save_jsonl(data, output_path):
    print(f"Saving {len(data)} records to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

# ==========================================
# 2. MAIN WORKFLOW
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Run BM25 Retrieval -> ReaderLLM -> Classification Report.")

    # Input Files
    parser.add_argument("--index_dir", required=True, help="Path to Pyserini Index")
    parser.add_argument("--segmented_queries", required=True, help="TSV File: id <tab> segmented_query")
    parser.add_argument("--raw_queries", required=True, help="CSV File: query_id, query (Natural Language)")
    parser.add_argument("--doc_mapping", required=True, help="CSV File: doc_id, document (Raw Text)")
    parser.add_argument("--labels", required=True, help="CSV Labels file (vifc_test_set_with_labels.csv)")
    
    # Config
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--top_k", type=int, default=3, help="Docs to retrieve for LLM")
    parser.add_argument("--api_key", type=str, help="Gemini API Key")
    parser.add_argument("--model_name", type=str, default="gemini-2.5-flash")
    parser.add_argument("--qid", type=str, help="Specific Query ID to process (Test Mode)")

    args = parser.parse_args()
    
    # 1. Setup
    api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key: sys.exit("Error: No API Key provided.")
    os.makedirs(args.output_dir, exist_ok=True)

    # 2. Initialize Models
    print(f">>> Initializing ReaderLLM ({args.model_name})...")
    reader = ReaderLLM(api_key=api_key, model_name=args.model_name)

    print(f">>> Initializing Searcher from {args.index_dir}...")
    if not os.path.exists(args.index_dir):
        sys.exit(f"Index not found at {args.index_dir}")
        
    searcher = LuceneSearcher(args.index_dir)
    searcher.set_bm25(k1=1.2, b=0.75)
    
    # Attempt to set Whitespace Analyzer (consistent with index build)
    try:
        JWhitespaceAnalyzer = autoclass('org.apache.lucene.analysis.core.WhitespaceAnalyzer')
        searcher.set_analyzer(JWhitespaceAnalyzer())
        print(">>> WhitespaceAnalyzer set.")
    except Exception as e:
        print(f"Warning: Could not set WhitespaceAnalyzer: {e}")

    # 3. Load Data
    seg_queries = load_segmented_queries(args.segmented_queries)
    raw_queries = load_raw_queries(args.raw_queries)
    doc_map = load_doc_mapping(args.doc_mapping)
    rag_labels = load_labels(args.labels, raw_queries)

    # 4. Filter Targets
    if args.qid:
        if args.qid not in seg_queries:
            print(f"Error: Query ID {args.qid} not found in segmented queries.")
            sys.exit(1)
        target_queries = {args.qid: seg_queries[args.qid]}
        print(f">>> TEST MODE: Processing single Query ID: {args.qid}")
    else:
        target_queries = seg_queries
        print(f">>> FULL MODE: Processing {len(target_queries)} queries...")

    # 5. Processing Loop
    all_predictions = []
    topk_retrievals = []

    for qid, seg_query in tqdm(target_queries.items(), desc="BM25+RAG"):
        # --- A. Retrieval (BM25) ---
        try:
            hits = searcher.search(seg_query, k=args.top_k)
        except Exception as e:
            print(f"Search error for QID {qid}: {e}")
            hits = []

        retrieved_ids = []
        llm_docs = []

        # --- B. Prepare Context ---
        for hit in hits:
            doc_id = hit.docid
            retrieved_ids.append(doc_id)
            
            # Prefer raw text from map, fallback to index
            content = doc_map.get(doc_id)
            if not content: 
                try: content = hit.raw
                except: pass
            
            if content:
                llm_docs.append({
                    "id": doc_id,
                    "content": content,
                    "score": float(hit.score)
                })

        topk_retrievals.append({
            "qid": qid,
            "topk_document_ids": retrieved_ids
        })

        if not llm_docs:
            print(f"Warning: No documents found for QID {qid}")
            continue

        # --- C. Generate Answer ---
        raw_q_text = raw_queries.get(qid, seg_query)
        
        # Call LLM
        raw_answer = reader.generate_answer(qid, raw_q_text, llm_docs)
        
        # Parse Output
        parsed = parse_llm_output(raw_answer)
        if isinstance(parsed, dict):
            if 'qid' not in parsed: parsed['qid'] = qid
            all_predictions.append(parsed)

    # ==========================================
    # 6. FINAL RESULTS
    # ==========================================
    
    save_jsonl(all_predictions, os.path.join(args.output_dir, "predictions.jsonl"))
    save_jsonl(topk_retrievals, os.path.join(args.output_dir, "topk_retrieval.jsonl"))

    print("\n" + "="*40)
    print("      EVALUATION RESULTS           ")
    print("="*40)
    
    y_true, y_pred = [], []
    for p in all_predictions:
        qid = str(p.get('qid'))
        verdict = p.get('verdict')
        if qid in rag_labels and verdict:
            y_true.append(rag_labels[qid])
            y_pred.append(verdict)

    if y_true:
        print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
        print("-" * 40)
        print("Detailed Classification Report:")
        print(classification_report(y_true, y_pred, zero_division=0))
    else:
        print("No labels matched for predicted queries (or single query not labeled).")

if __name__ == "__main__":
    main()
