import argparse
from collections import defaultdict
import os
from tqdm import tqdm

def read_run_file(filepath):
    """
    Reads a run file (qid, docid, rank, score).
    Returns: dict {qid: {docid: score}}
    """
    run_data = defaultdict(dict)
    print(f"Reading {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            # Handle potential variation in whitespace splitting vs tab splitting
            if len(parts) < 4:
                parts = line.strip().split()

            if len(parts) < 4:
                continue

            # Assuming format: qid docid rank score (or qid Q0 docid rank score runtag)
            # If 4 columns: qid, docid, rank, score
            if len(parts) == 4:
                qid, docid, _, score = parts
            # If standard TREC 6 columns: qid, Q0, docid, rank, score, tag
            elif len(parts) >= 6:
                qid = parts[0]
                docid = parts[2]
                score = parts[4]
            else:
                # Fallback to index 0, 1, 3 based on typical previous notebook output
                qid = parts[0]
                docid = parts[1]
                score = parts[3]

            try:
                run_data[qid][docid] = float(score)
            except ValueError:
                continue

    return run_data

def normalize_scores(run_data):
    """
    Min-Max normalization per query.
    Maps scores to [0, 1] range based on the min and max score for that query.
    """
    normalized = defaultdict(dict)
    for qid, docs in run_data.items():
        if not docs:
            continue

        scores = list(docs.values())
        min_s = min(scores)
        max_s = max(scores)
        diff = max_s - min_s

        if diff == 0:
            # If all scores are the same, set them to 1.0
            for docid in docs: 
                normalized[qid][docid] = 1.0
        else:
            for docid, score in docs.items():
                normalized[qid][docid] = (score - min_s) / diff
    return normalized

def main():
    parser = argparse.ArgumentParser(description="Create a hybrid run file from BM25 and DeeperImpact runs.")
    parser.add_argument("--bm25_run", required=True, help="Path to the BM25 run file")
    parser.add_argument("--deep_impact_run", required=True, help="Path to the DeeperImpact run file")
    parser.add_argument("--output_file", required=True, help="Path to save the generated hybrid run file")
    parser.add_argument("--alpha", type=float, required=True, help="Weight for BM25 (0.3 to 0.7). Formula: score = alpha * bm25 + (1-alpha) * deep_impact")
    parser.add_argument("--top_k", type=int, default=1000, help="Number of documents to keep per query in the output")
    parser.add_argument("--normalize", action="store_true", help="Enable Min-Max normalization for scores before combination")

    args = parser.parse_args()

    # 1. Load Runs
    bm25_data = read_run_file(args.bm25_run)
    di_data = read_run_file(args.deep_impact_run)

    # 2. Normalize (Optional)
    if args.normalize:
        print("Normalizing scores (Min-Max)...")
        bm25_data = normalize_scores(bm25_data)
        di_data = normalize_scores(di_data)

    # 3. Combine Scores
    print(f"Calculating Hybrid Scores with alpha={args.alpha}...")
    hybrid_results = defaultdict(dict)

    # Get all unique QIDs from both runs
    all_qids = set(bm25_data.keys()) | set(di_data.keys())

    for qid in tqdm(all_qids, desc="Processing queries"):
        # Get union of all documents retrieved by either method for this query
        all_docs = set(bm25_data.get(qid, {}).keys()) | set(di_data.get(qid, {}).keys())

        for docid in all_docs:
            # Get scores, defaulting to 0.0 if not found
            # Note: If using normalization, 0.0 corresponds to the minimum score in the list
            s_bm25 = bm25_data.get(qid, {}).get(docid, 0.0)
            s_di = di_data.get(qid, {}).get(docid, 0.0)

            # Hybrid Formula
            s_hybrid = (s_bm25 * args.alpha) + ((1.0 - args.alpha) * s_di)

            hybrid_results[qid][docid] = s_hybrid

    # 4. Write Output
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    print(f"Writing results to {args.output_file}...")

    with open(args.output_file, 'w', encoding='utf-8') as f:
        # Sort queries (numerically if possible for cleaner output)
        sorted_qids = sorted(hybrid_results.keys(), key=lambda x: int(x) if x.isdigit() else x)

        for qid in sorted_qids:
            # Sort docs by descending hybrid score
            sorted_docs = sorted(hybrid_results[qid].items(), key=lambda item: item[1], reverse=True)[:args.top_k]

            for rank, (docid, score) in enumerate(sorted_docs, start=1):
                # qid \t docid \t rank \t score
                f.write(f"{qid}\t{docid}\t{rank}\t{score:.6f}\n")

    print("Hybrid run file created successfully.")

if __name__ == "__main__":
    main()
