import json
import csv
import argparse
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def load_queries(query_mapping_path):
    """Loads QueryID -> Query Text mapping."""
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
    Loads labels from CSV and maps them to QIDs.
    Logic mirrors rag_inference.py to ensure consistency.
    """
    print(f"Loading labels from {label_file_path}...")
    try:
        df = pd.read_csv(label_file_path)
    except Exception as e:
        print(f"Error reading label file: {e}")
        return {}

    # Map Query Text -> Label
    query_text_to_label = dict(zip(df['query'].str.strip(), df['label'].str.strip()))

    qid_to_label = {}
    for qid, q_text in queries_map.items():
        clean_text = q_text.strip()
        if clean_text in query_text_to_label:
            qid_to_label[qid] = query_text_to_label[clean_text]

    print(f"Mapped {len(qid_to_label)} ground truth labels.")
    return qid_to_label

def load_predictions(pred_file_path):
    """Loads QID -> Verdict from predictions.jsonl."""
    print(f"Loading predictions from {pred_file_path}...")
    preds = {}
    with open(pred_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                entry = json.loads(line)
                qid = str(entry.get('qid', '')).strip()
                verdict = entry.get('verdict')
                if qid and verdict:
                    preds[qid] = verdict
            except json.JSONDecodeError:
                continue
    return preds

def load_group_qids(jsonl_path):
    """Extracts QIDs from the split group files (high/low overlap)."""
    qids = set()
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line)
                if 'qid' in entry:
                    qids.add(str(entry['qid']).strip())
    except FileNotFoundError:
        print(f"Warning: Group file {jsonl_path} not found.")
    return qids

def calculate_metrics(name, qids_in_group, all_preds, all_labels):
    """Calculates metrics for a specific subset of QIDs."""
    y_true = []
    y_pred = []

    for qid in qids_in_group:
        # We need both a prediction and a label to evaluate
        if qid in all_preds and qid in all_labels:
            y_true.append(all_labels[qid])
            y_pred.append(all_preds[qid])

    if not y_true:
        return {"name": name, "count": 0, "accuracy": 0, "f1": 0, "precision": 0, "recall": 0}

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )

    return {
        "name": name,
        "count": len(y_true),
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

def main():
    parser = argparse.ArgumentParser(description="Compare F1 scores between High and Low overlap groups.")

    # Files
    parser.add_argument("--predictions", required=True, help="Path to predictions.jsonl")
    parser.add_argument("--labels", required=True, help="Path to vifc_test_set_with_labels.csv")
    parser.add_argument("--query_mapping", required=True, help="Path to test_query_mapping.csv")
    parser.add_argument("--high_group", required=True, help="Path to high_overlap_queries.jsonl")
    parser.add_argument("--low_group", required=True, help="Path to low_overlap_queries.jsonl")

    args = parser.parse_args()

    # 1. Load Data
    queries_map = load_queries(args.query_mapping)
    ground_truth = load_ground_truth(args.labels, queries_map)
    predictions = load_predictions(args.predictions)

    # 2. Load Groups
    high_qids = load_group_qids(args.high_group)
    low_qids = load_group_qids(args.low_group)

    print(f"\nFound {len(high_qids)} queries in High Group.")
    print(f"Found {len(low_qids)} queries in Low Group.")

    # 3. Calculate Metrics
    high_metrics = calculate_metrics("High Overlap", high_qids, predictions, ground_truth)
    low_metrics = calculate_metrics("Low Overlap", low_qids, predictions, ground_truth)

    # 4. Print Comparison Table
    print("\n" + "="*65)
    print(f"{'METRIC':<15} | {'HIGH OVERLAP GROUP':<20} | {'LOW OVERLAP GROUP':<20}")
    print("="*65)

    metrics_list = [
        ("Count (Eval)", "count", "{:d}"),
        ("Accuracy", "accuracy", "{:.4f}"),
        ("F1-Score", "f1", "{:.4f}"),
        ("Precision", "precision", "{:.4f}"),
        ("Recall", "recall", "{:.4f}")
    ]

    for label, key, fmt in metrics_list:
        val_high = fmt.format(high_metrics[key])
        val_low = fmt.format(low_metrics[key])
        print(f"{label:<15} | {val_high:<20} | {val_low:<20}")

    print("="*65)

    # 5. Interpretation Helper
    f1_diff = high_metrics['f1'] - low_metrics['f1']
    print(f"\nAnalysis: High Overlap group performs {abs(f1_diff)*100:.2f}% {'better' if f1_diff > 0 else 'worse'} than Low Overlap group.")

if __name__ == "__main__":
    main()
