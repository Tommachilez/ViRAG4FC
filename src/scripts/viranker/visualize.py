import json
import logging
import os
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import CrossEncoder
from sklearn.model_selection import train_test_split

# --- Configuration ---
BASE_MODEL_NAME = "BAAI/bge-m3"
TRAINED_MODEL_PATH = "./viranker_checkpoint"
DATA_FILE = "train_data.jsonl"
DEV_SPLIT_RATIO = 0.1
K_VALUES = [3, 5, 10]
OUTPUT_IMAGE = "viranker_benchmark.png"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# --- Metric Calculation (Reused Logic) ---

def compute_ndcg_at_k(relevance_scores, k):
    """Calculate NDCG@k for a single query."""
    relevance_scores = np.asfarray(relevance_scores)[:k]
    if relevance_scores.size == 0: return 0.0

    # DCG
    dcg = np.sum(relevance_scores / np.log2(np.arange(2, relevance_scores.size + 2)))

    # IDCG
    ideal_relevance = sorted(relevance_scores, reverse=True)
    ideal_relevance = np.asfarray(ideal_relevance)[:k]
    idcg = np.sum(ideal_relevance / np.log2(np.arange(2, ideal_relevance.size + 2)))

    return dcg / idcg if idcg > 0 else 0.0

def compute_mrr_at_k(relevance_scores, k):
    """Calculate MRR@k for a single query."""
    for i, score in enumerate(relevance_scores[:k]):
        if score > 0: return 1.0 / (i + 1)
    return 0.0

def evaluate_model(model_path, dev_data):
    """
    Loads a model and evaluates it against the dev set.
    Returns a dictionary of averaged scores.
    """
    logging.info(f"Loading model: {model_path} ...")
    try:
        model = CrossEncoder(model_path, max_length=512)
    except Exception as e:
        logging.error(f"Failed to load {model_path}: {e}")
        return None

    logging.info(f"Evaluating on {len(dev_data)} queries...")

    # Store all scores
    scores = {f"NDCG@{k}": [] for k in K_VALUES}
    scores.update({f"MRR@{k}": [] for k in K_VALUES})

    for entry in dev_data:
        query = entry['query']
        pos_docs = entry['pos']
        neg_docs = entry['neg']

        all_docs = pos_docs + neg_docs
        # Labels: 1 for pos, 0 for neg
        labels = [1] * len(pos_docs) + [0] * len(neg_docs)

        if not all_docs: continue

        pairs = [[query, doc] for doc in all_docs]
        pred_scores = model.predict(pairs)

        # Sort labels by predicted score descending
        ranked_results = sorted(zip(labels, pred_scores), key=lambda x: x[1], reverse=True)
        ranked_labels = [x[0] for x in ranked_results]

        for k in K_VALUES:
            scores[f"NDCG@{k}"].append(compute_ndcg_at_k(ranked_labels, k))
            scores[f"MRR@{k}"].append(compute_mrr_at_k(ranked_labels, k))

    # Calculate Averages
    avg_scores = {k: np.mean(v) for k, v in scores.items()}
    return avg_scores

# --- Plotting ---

def plot_comparison(base_scores, trained_scores):
    """Generates a side-by-side bar chart."""
    metrics = list(base_scores.keys())

    # Sort metrics so MRR and NDCG are grouped logically
    metrics.sort(key=lambda x: (int(x.split('@')[1]), x.split('@')[0]))

    base_vals = [base_scores[m] for m in metrics]
    trained_vals = [trained_scores[m] for m in metrics]

    x = np.arange(len(metrics))  # label locations
    width = 0.35  # width of the bars

    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, base_vals, width, label='Base Model', color='#95a5a6')
    rects2 = ax.bar(x + width/2, trained_vals, width, label='ViRanker (Trained)', color='#2ecc71')

    # Add text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Score')
    ax.set_title('Performance Comparison: Base vs Trained ViRanker')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Add value labels on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    plt.savefig(OUTPUT_IMAGE)
    logging.info(f"Comparison chart saved to {OUTPUT_IMAGE}")
    plt.show()

# --- Main Execution ---

def main():
    # 1. Load Data
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found. Please run training first or provide data.")
        return

    all_data = []
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            try: all_data.append(json.loads(line))
            except: continue

    # Use exact same random_state=42 to match training split
    _, dev_data = train_test_split(all_data, test_size=DEV_SPLIT_RATIO, random_state=42)

    # 2. Evaluate Base Model
    logging.info("--- Evaluating Base Model ---")
    base_results = evaluate_model(BASE_MODEL_NAME, dev_data)

    # 3. Evaluate Trained Model
    logging.info("--- Evaluating Trained Model ---")
    if not os.path.exists(TRAINED_MODEL_PATH):
        logging.warning("Trained model path not found. Plotting Base model only (or check path).")
        trained_results = {k: 0.0 for k in base_results}
    else:
        trained_results = evaluate_model(TRAINED_MODEL_PATH, dev_data)

    # 4. Visualize
    if base_results and trained_results:
        print("\nResults Summary:")
        print(f"{'Metric':<10} | {'Base':<10} | {'Trained':<10} | {'Gain':<10}")
        print("-" * 46)
        for m in sorted(base_results.keys()):
            gain = ((trained_results[m] - base_results[m]) / base_results[m]) * 100 if base_results[m] > 0 else 0
            print(f"{m:<10} | {base_results[m]:.4f}     | {trained_results[m]:.4f}     | {gain:+.1f}%")

        plot_comparison(base_results, trained_results)

if __name__ == "__main__":
    main()
