import logging
import numpy as np


class RankerEvaluator:
    """
    Custom evaluator to calculate NDCG@k and MRR@k for reranking tasks.
    """
    def __init__(self, dev_data, k_values=None):
        self.dev_data = dev_data # List of {"query": str, "pos": [], "neg": []}
        self.k_values = k_values

    def compute_dcg_at_k(self, relevance_scores, k):
        relevance_scores = np.asfarray(relevance_scores)[:k]
        if relevance_scores.size == 0:
            return 0.0
        # DCG formula: sum(rel_i / log2(i + 2)) -- using i+2 because index starts at 0
        return np.sum(relevance_scores / np.log2(np.arange(2, relevance_scores.size + 2)))

    def compute_ndcg_at_k(self, relevance_scores, k):
        dcg = self.compute_dcg_at_k(relevance_scores, k)

        # IDCG: Sort relevance scores descending (ideal case)
        ideal_relevance = sorted(relevance_scores, reverse=True)
        idcg = self.compute_dcg_at_k(ideal_relevance, k)

        return dcg / idcg if idcg > 0 else 0.0

    def compute_mrr_at_k(self, relevance_scores, k):
        # We assume relevance_scores are binary (1 for pos, 0 for neg) for MRR
        for i, score in enumerate(relevance_scores[:k]):
            if score > 0: # Found the first relevant document
                return 1.0 / (i + 1)
        return 0.0

    def __call__(self, model, epoch_idx=0, steps=0):
        """
        This method is called by the training loop or manually to evaluate.
        """
        logging.info(f"Starting evaluation on {len(self.dev_data)} queries...")

        ndcg_scores = {k: [] for k in self.k_values}
        mrr_scores = {k: [] for k in self.k_values}

        for entry in self.dev_data:
            query = entry['query']
            pos_docs = entry['pos']
            neg_docs = entry['neg']

            # Combine docs and ground truth labels
            # Label 1 for Positive, 0 for Negative
            all_docs = pos_docs + neg_docs
            labels = [1] * len(pos_docs) + [0] * len(neg_docs)

            if not all_docs:
                continue

            # Prepare pairs for the model
            pairs = [[query, doc] for doc in all_docs]

            # Predict scores using the CrossEncoder
            pred_scores = model.predict(pairs)

            # Zip scores with labels and sort by predicted score (Descending)
            ranked_results = sorted(zip(labels, pred_scores), key=lambda x: x[1], reverse=True)

            # Extract the ranked relevance labels (e.g., [1, 0, 1, 0...])
            ranked_labels = [x[0] for x in ranked_results]

            # Calculate metrics for each k
            for k in self.k_values:
                ndcg_scores[k].append(self.compute_ndcg_at_k(ranked_labels, k))
                mrr_scores[k].append(self.compute_mrr_at_k(ranked_labels, k))

        # Aggregate and Print
        print("\n" + "="*40)
        print(f"EVALUATION RESULTS (Epoch {epoch_idx})")
        print("="*40)
        for k in self.k_values:
            avg_ndcg = np.mean(ndcg_scores[k])
            avg_mrr = np.mean(mrr_scores[k])
            print(f"NDCG@{k:<2}: {avg_ndcg:.4f} | MRR@{k:<2}: {avg_mrr:.4f}")
        print("="*40 + "\n")
