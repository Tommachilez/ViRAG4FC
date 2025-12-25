import sys
import json
import logging
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import CrossEncoder
from tqdm import tqdm

# --- Configuration Constants (Defaults) ---
DEFAULT_BASE_MODEL = "namdp-ptit/ViRanker"
DEFAULT_TRAINED_MODEL = "./viranker_checkpoint"
DEFAULT_DATA_FILE = "train_data.jsonl"
DEFAULT_OUTPUT_IMG = "viranker_benchmark.png"
K_VALUES = [3, 5, 10]

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate and Compare ViRanker Models with MaxP Support")

    # Model Paths
    parser.add_argument("--base_model", type=str, default=DEFAULT_BASE_MODEL, help="HuggingFace model ID or path for the Baseline model.")
    parser.add_argument("--trained_model", type=str, default=DEFAULT_TRAINED_MODEL, help="Path to the trained checkpoint.")

    # Data & Output
    parser.add_argument("--jsonl", type=str, default=DEFAULT_DATA_FILE, help="Path to the training data JSONL file.")
    parser.add_argument("--output_image", type=str, default=DEFAULT_OUTPUT_IMG, help="Filename for the comparison chart.")

    # MaxP / Evaluation Settings
    parser.add_argument("--maxp", action="store_true", help="Enable MaxP sliding window scoring.")
    parser.add_argument("--window_size", type=int, default=250, help="Window size in words for MaxP (default: 250).")
    parser.add_argument("--stride", type=int, default=100, help="Stride in words for MaxP (default: 100).")
    parser.add_argument("--max_length", type=int, default=1024, help="Max sequence length for the CrossEncoder (default: 1024).")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference.")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda/cpu). Default is auto.")

    return parser.parse_args()

# --- Helper: Sliding Window ---
def sliding_window(text, window_size=250, stride=100):
    """
    Splits text into overlapping chunks of words.
    """
    tokens = text.split()
    if not tokens:
        return [""]

    if len(tokens) <= window_size:
        return [text]

    windows = []
    for i in range(0, len(tokens), stride):
        chunk = " ".join(tokens[i : i + window_size])
        windows.append(chunk)
        if i + window_size >= len(tokens):
            break
    return windows

# --- Metric Calculation ---
def compute_ndcg_at_k(relevance_scores, k):
    """Calculate NDCG@k for a single query."""
    relevance_scores = np.asarray(relevance_scores, dtype=float)[:k]
    if relevance_scores.size == 0: return 0.0

    # DCG
    dcg = np.sum(relevance_scores / np.log2(np.arange(2, relevance_scores.size + 2)))

    # IDCG
    ideal_relevance = sorted(relevance_scores, reverse=True)
    ideal_relevance = np.asarray(ideal_relevance, dtype=float)[:k]
    idcg = np.sum(ideal_relevance / np.log2(np.arange(2, ideal_relevance.size + 2)))

    return dcg / idcg if idcg > 0 else 0.0

def compute_mrr_at_k(relevance_scores, k):
    """Calculate MRR@k for a single query."""
    for i, score in enumerate(relevance_scores[:k]):
        if score > 0: return 1.0 / (i + 1)
    return 0.0

# --- Evaluation Core ---
def evaluate_model(model_path_or_name, dev_data, args):
    """
    Loads a model and evaluates it against the dev set using args for configuration.
    """
    logging.info(f"Loading model: {model_path_or_name} (MaxP={args.maxp}) ...")

    device = args.device if args.device else None

    try:
        model = CrossEncoder(model_path_or_name, max_length=args.max_length, device=device)
    except Exception as e:
        logging.error(f"Failed to load {model_path_or_name}: {e}")
        return None

    logging.info(f"Evaluating on {len(dev_data)} queries...")

    # Store all scores
    scores = {f"NDCG@{k}": [] for k in K_VALUES}
    scores.update({f"MRR@{k}": [] for k in K_VALUES})

    for entry in tqdm(dev_data, desc="Evaluating"):
        try:
            query = entry['query']
            # Support both 'candidates' dict (from mining) or 'pos'/'neg' lists (from training prep)
            if 'pos' in entry and 'neg' in entry:
                pos_docs = entry['pos']
                neg_docs = entry['neg']
            elif 'candidates' in entry:
                cands = list(entry['candidates'].values())
                pos_docs = [cands[0]]
                neg_docs = cands[1:]
            else:
                continue

            all_docs = pos_docs + neg_docs
            # Labels: 1 for pos, 0 for neg
            labels = [1] * len(pos_docs) + [0] * len(neg_docs)

            if not all_docs: continue

            pred_scores = []

            # --- SCORING LOGIC ---
            if args.maxp:
                # MaxP: Loop through docs, split, score windows, take max
                for doc in all_docs:
                    windows = sliding_window(doc, window_size=args.window_size, stride=args.stride)
                    # Create pairs for this specific document's windows
                    window_pairs = [[query, w] for w in windows]

                    # Predict scores for all windows (batched internally by CrossEncoder)
                    w_scores = model.predict(window_pairs, batch_size=args.batch_size, show_progress_bar=False)

                    # Take the max score for this document
                    if isinstance(w_scores, (np.ndarray, list)):
                         # Handle single float return or list return
                        doc_score = np.max(w_scores)
                    else:
                        doc_score = w_scores

                    pred_scores.append(doc_score)
            else:
                # Standard (FirstP): Batch all docs at once for speed
                pairs = [[query, doc] for doc in all_docs]
                pred_scores = model.predict(pairs, batch_size=args.batch_size, show_progress_bar=False)

            # Sort labels by predicted score descending
            ranked_results = sorted(zip(labels, pred_scores), key=lambda x: x[1], reverse=True)
            ranked_labels = [x[0] for x in ranked_results]

            for k in K_VALUES:
                scores[f"NDCG@{k}"].append(compute_ndcg_at_k(ranked_labels, k))
                scores[f"MRR@{k}"].append(compute_mrr_at_k(ranked_labels, k))

        except Exception as e:
            # logging.warning(f"Skipping query due to error: {e}")
            continue

    # Calculate Averages
    avg_scores = {k: np.mean(v) for k, v in scores.items()}
    return avg_scores

# --- Plotting ---
def plot_comparison(base_scores, trained_scores, output_path):
    """Generates a side-by-side bar chart."""
    metrics = list(base_scores.keys())
    # Sort metrics so MRR and NDCG are grouped logically
    metrics.sort(key=lambda x: (int(x.split('@')[1]), x.split('@')[0]))

    base_vals = [base_scores[m] for m in metrics]
    trained_vals = [trained_scores[m] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, base_vals, width, label='Base Model', color='#95a5a6')
    rects2 = ax.bar(x + width/2, trained_vals, width, label='Trained Model', color='#2ecc71')

    ax.set_ylabel('Score')
    ax.set_title('Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    plt.savefig(output_path)
    logging.info(f"Comparison chart saved to {output_path}")
    plt.show() # Uncomment if running locally with GUI

# --- Main Execution ---
def main():
    args = parse_args()

    # 1. Load Data
    if not os.path.exists(args.jsonl):
        logging.error(f"Data file {args.jsonl} not found.")
        return

    print(f"Loading data from {args.jsonl}...")
    dev_data = [] # Treat all data as dev data
    with open(args.jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            try: dev_data.append(json.loads(line))
            except: continue

    print(f"Total queries for evaluation: {len(dev_data)}")

    # 2. Evaluate Trained Model
    print(f"--- Evaluating Trained Model: {args.trained_model} ---")
    if not os.path.exists(args.trained_model) and not args.trained_model.startswith("namdp-ptit"):
        logging.warning(f"Trained model path '{args.trained_model}' not found on disk. Comparing against 0s.")
        sys.exit(0)
    else:
        trained_results = evaluate_model(args.trained_model, dev_data, args)

    # 3. Evaluate Base Model
    print(f"--- Evaluating Base Model: {args.base_model} ---")
    base_results = evaluate_model(args.base_model, dev_data, args)

    # 4. Visualize
    if base_results and trained_results:
        print("\n" + "="*50)
        print("RESULTS SUMMARY")
        print("="*50)
        print(f"{'Metric':<10} | {'Base':<10} | {'Trained':<10} | {'Gain':<10}")
        print("-" * 50)

        metrics = sorted(base_results.keys(), key=lambda x: (int(x.split('@')[1]), x.split('@')[0]))

        for m in metrics:
            b_val = base_results[m]
            t_val = trained_results[m]
            gain = ((t_val - b_val) / b_val) * 100 if b_val > 0 else 0.0
            print(f"{m:<10} | {b_val:.4f}     | {t_val:.4f}     | {gain:+.1f}%")
        print("="*50 + "\n")

        plot_comparison(base_results, trained_results, args.output_image)

if __name__ == "__main__":
    main()
