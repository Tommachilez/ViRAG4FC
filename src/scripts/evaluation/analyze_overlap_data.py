import json
import argparse
import pandas as pd
import sys
from pathlib import Path
from typing import Dict, Any

def analyze_overlap_data(input_path: str, split_method: str = 'median', custom_threshold: float = None, aggregation: str = 'max') -> Dict[str, Any]:
    """
    Args:
        aggregation (str): 'max' (best document) or 'mean' (average of all).
    """
    data = []

    print(f"Loading data from {input_path}...")
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    qid = entry.get('qid')
                    scores_map = entry.get('document_overlap_scores', {})

                    if not scores_map:
                        final_score = 0.0
                    else:
                        scores = list(scores_map.values())
                        if aggregation == 'max':
                            final_score = max(scores)
                        else:
                            final_score = sum(scores) / len(scores)

                    data.append({
                        "qid": qid,
                        "score": final_score, # Renamed from avg_score to generic 'score'
                        "raw_scores": scores_map 
                    })
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        print(f"Error: File {input_path} not found.")
        sys.exit(1)

    if not data:
        print("No valid data found.")
        return {}

    df = pd.DataFrame(data)

    # Determine Statistics
    stats = df['score'].describe(percentiles=[0.25, 0.5, 0.75]).to_dict()

    # Determine Split Threshold
    if split_method == 'custom' and custom_threshold is not None:
        threshold = custom_threshold
    elif split_method == 'mean':
        threshold = stats['mean']
    else: # default to median
        threshold = stats['50%']

    # Split Data
    high_group = df[df['score'] >= threshold].copy()
    low_group = df[df['score'] < threshold].copy()

    # Sort
    high_group = high_group.sort_values(by='score', ascending=False)
    low_group = low_group.sort_values(by='score', ascending=True)

    return {
        "statistics": stats,
        "threshold_used": threshold,
        "high_overlap": high_group,
        "low_overlap": low_group
    }

def save_split_results(df: pd.DataFrame, output_path: str):
    records = df.to_dict(orient='records')
    with open(output_path, 'w', encoding='utf-8') as f:
        for record in records:
            out_obj = {
                "qid": record['qid'],
                "aggregated_score": round(record['score'], 4), # Renamed for clarity
                "document_overlap_scores": record['raw_scores']
            }
            f.write(json.dumps(out_obj, ensure_ascii=False) + "\n")

def main():
    parser = argparse.ArgumentParser(description="Analyze overlap scores and split queries.")
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_dir", required=True)

    # New Argument for Aggregation
    parser.add_argument("--aggregation", choices=['max', 'mean'], default='max', 
                        help="How to combine scores from top-k docs (max=best doc, mean=average quality).")

    parser.add_argument("--split_method", choices=['median', 'mean', 'custom'], default='median')
    parser.add_argument("--threshold", type=float)

    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Aggregating scores using: {args.aggregation.upper()}")

    results = analyze_overlap_data(args.input_file, args.split_method, args.threshold, args.aggregation)

    if not results: return

    stats = results['statistics']
    threshold = results['threshold_used']

    # Print Statistics to Console
    print("\n" + "="*40)
    print(" OVERLAP SCORE STATISTICS")
    print("="*40)
    print(f"Total Queries: {int(stats['count'])}")
    print(f"Min Score:     {stats['min']:.4f}")
    print(f"Max Score:     {stats['max']:.4f}")
    print(f"Mean:          {stats['mean']:.4f}")
    print("-" * 40)
    print("Quantiles:")
    print(f"  25% (Q1):    {stats['25%']:.4f}")
    print(f"  50% (Median):{stats['50%']:.4f}")
    print(f"  75% (Q3):    {stats['75%']:.4f}")
    print("="*40)
    print(f"Splitting data based on {args.split_method.upper()} threshold: {threshold:.4f} (with {args.aggregation} aggregation)")

    # Save files (omitted boilerplate, same as previous)
    high_path = out_dir / "high_overlap_queries.jsonl"
    low_path = out_dir / "low_overlap_queries.jsonl"
    save_split_results(results['high_overlap'], str(high_path))
    save_split_results(results['low_overlap'], str(low_path))
    print(f"Saved results to {out_dir}")

if __name__ == "__main__":
    main()
