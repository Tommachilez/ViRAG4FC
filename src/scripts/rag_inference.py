import csv
import argparse
import os
import json
import ast
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from .reader_llm import ReaderLLM

def load_document_mapping(doc_mapping_path):
    """Loads the map of DocID -> Document Text."""
    print(f"Loading document text from {doc_mapping_path}...")
    doc_map = {}
    with open(doc_mapping_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc="Loading Docs"):
            if 'doc_id' in row and 'document' in row:
                doc_map[row['doc_id'].strip()] = row['document'].strip()
    return doc_map

def load_queries(query_mapping_path):
    """Loads the map of QueryID -> Query Text."""
    print(f"Loading queries from {query_mapping_path}...")
    queries = {}
    with open(query_mapping_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if 'query_id' in row and 'query' in row:
                queries[row['query_id'].strip()] = row['query'].strip()
    return queries

def load_ground_truth(label_file_path, queries_map):
    """
    Loads ground truth labels from the CSV file.
    Maps Query Text -> Label, then resolves to QID -> Label using queries_map.
    """
    print(f"Loading labels from {label_file_path}...")
    try:
        df = pd.read_csv(label_file_path)
    except Exception as e:
        print(f"Error reading label file: {e}")
        return {}

    # Create map: Query Text -> Label
    # We strip whitespace to ensure better matching
    query_text_to_label = dict(zip(df['query'].str.strip(), df['label'].str.strip()))

    qid_to_label = {}
    for qid, q_text in queries_map.items():
        clean_text = q_text.strip()
        if clean_text in query_text_to_label:
            qid_to_label[qid] = query_text_to_label[clean_text]

    print(f"Mapped {len(qid_to_label)} ground truth labels to Query IDs.")
    return qid_to_label

def load_run_file(run_file_path, top_k=5):
    """
    Parses the DeeperImpact run file to get top-K documents per query.
    """
    print(f"Loading top-{top_k} results from {run_file_path}...")
    retrieved_results = defaultdict(list)

    with open(run_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 4: continue

            qid = parts[0]
            doc_id = parts[1]
            rank = int(parts[2])
            score = float(parts[3])

            if rank <= top_k:
                retrieved_results[qid].append((doc_id, score))

    return retrieved_results

def save_jsonl(data, output_path):
    """Helper to save a list of dicts to a JSONL file."""
    print(f"Saving {len(data)} records to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

def parse_llm_output(output_str):
    """
    Robustly parses LLM output into a dictionary.
    """
    # 1. If it's already a dict, return it immediately (Fixes your TypeError)
    if isinstance(output_str, dict):
        return output_str

    # 2. If it's a string, try parsing as standard JSON
    try:
        return json.loads(output_str)
    except (json.JSONDecodeError, TypeError):
        pass

    # 3. Fallback: Try parsing as a Python dictionary string (e.g., "{'key': 'val'}")
    try:
        return ast.literal_eval(output_str)
    except (ValueError, SyntaxError):
        pass

    # 4. If all else fails, return an error dict with the raw text
    return {
        "qid": None,
        "verdict": "Error",
        "explanation": f"Failed to parse output: {str(output_str)[:100]}..."
    }

def calculate_metrics(predictions, ground_truth_map):
    """
    Calculates Accuracy, Precision, Recall, and F1.
    """
    y_true = []
    y_pred = []

    for pred in predictions:
        qid = pred.get('qid')
        verdict = pred.get('verdict')

        # Only evaluate if we have a ground truth label and a valid prediction
        if qid in ground_truth_map and verdict:
            y_true.append(ground_truth_map[qid])
            y_pred.append(verdict)

    if not y_true:
        print("No matching ground truth labels found for the processed queries.")
        return

    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("-" * 60)
    print("Detailed Classification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))
    print("="*60 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Run RAG using DeeperImpact results and ReaderLLM.")

    # Data Inputs
    parser.add_argument("--run_file", type=str, required=True, help="Path to run_file.txt")
    parser.add_argument("--doc_mapping", type=str, required=True, help="Path to unique_doc_mapping.csv")
    parser.add_argument("--query_mapping", type=str, required=True, help="Path to test_query_mapping.csv")
    parser.add_argument("--label_file", type=str, required=True, help="Path to vifc_test_set_with_labels.csv")

    # Execution Config
    parser.add_argument("--qid", type=str, help="Specific Query ID to process (optional)")
    parser.add_argument("--top_k", type=int, default=3, help="Number of documents to send to LLM")

    # Output Config
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output JSONL files")

    # LLM Config
    parser.add_argument("--api_key", type=str, help="Gemini API Key")
    parser.add_argument("--model_name", type=str, default="gemini-2.5-flash", help="Gemini model name")

    args = parser.parse_args()

    # Determine API Key
    api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: API Key must be provided via --api_key or env var.")
        exit(1)

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Initialize Reader LLM
    print(f">>> Initializing Reader LLM with model: {args.model_name}...")
    try:
        reader = ReaderLLM(api_key=api_key, model_name=args.model_name)
    except Exception as e:
        print(f"Error initializing Gemini: {e}")
        exit(1)

    # 2. Load Data
    doc_map = load_document_mapping(args.doc_mapping)
    queries = load_queries(args.query_mapping)
    run_data = load_run_file(args.run_file, top_k=args.top_k)
    ground_truth = load_ground_truth(args.label_file, queries)

    # 3. Determine targets
    target_qids = [args.qid] if args.qid else list(queries.keys())

    # Storage for outputs
    all_predictions = []
    topk_retrievals = []

    # 4. RAG Execution Loop
    for qid in tqdm(target_qids, desc="Processing Queries"):
        qid = str(qid)
        if qid not in queries:
            continue

        query_text = queries[qid]
        retrieved_docs_ids = run_data.get(qid, [])

        # Store Retrieval Info
        topk_retrievals.append({
            "qid": qid,
            "topk_document_ids": [doc[0] for doc in retrieved_docs_ids]
        })

        # Format docs for ReaderLLM
        llm_input_docs = []
        for doc_id, score in retrieved_docs_ids:
            content = doc_map.get(doc_id)
            if content:
                llm_input_docs.append({
                    "id": doc_id,
                    "content": content,
                    "score": score
                })

        if not llm_input_docs:
            continue

        # Generate Answer
        raw_answer = reader.generate_answer(qid, query_text, llm_input_docs)

        # Parse Answer
        parsed_answer = parse_llm_output(raw_answer)

        # If parsing returned a dict, ensure QID is preserved
        if isinstance(parsed_answer, dict):
            if 'qid' not in parsed_answer:
                parsed_answer['qid'] = qid
            all_predictions.append(parsed_answer)

    # 5. Save Outputs
    pred_file = os.path.join(args.output_dir, "predictions.jsonl")
    retrieval_file = os.path.join(args.output_dir, "topk_retrieval.jsonl")

    save_jsonl(all_predictions, pred_file)
    save_jsonl(topk_retrievals, retrieval_file)

    # 6. Calculate Metrics
    if all_predictions:
        calculate_metrics(all_predictions, ground_truth)
    else:
        print("No predictions were generated.")

if __name__ == "__main__":
    main()
