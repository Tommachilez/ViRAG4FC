import numpy as np
from tqdm import tqdm


class RankerEvaluator:
    """
    Custom evaluator to calculate NDCG@k and MRR@k for reranking tasks.
    """
    def __init__(self, dev_data, k_values=None):
        self.dev_data = dev_data
        self.k_values = k_values if k_values is not None else [3, 5, 10]

    def compute_dcg_at_k(self, relevance_scores, k):
        # FIX: np.asfarray is removed in NumPy 2.0 -> Use np.asarray with float dtype
        relevance_scores = np.asarray(relevance_scores, dtype=float)[:k]

        if relevance_scores.size == 0:
            return 0.0
        # DCG formula: sum(rel_i / log2(i + 2))
        return np.sum(relevance_scores / np.log2(np.arange(2, relevance_scores.size + 2)))

    def compute_ndcg_at_k(self, relevance_scores, k):
        dcg = self.compute_dcg_at_k(relevance_scores, k)
        # IDCG: Sort relevance scores descending (ideal case)
        ideal_relevance = sorted(relevance_scores, reverse=True)
        idcg = self.compute_dcg_at_k(ideal_relevance, k)
        return dcg / idcg if idcg > 0 else 0.0

    def compute_mrr_at_k(self, relevance_scores, k):
        # We assume relevance_scores are binary (1 for pos, 0 for neg)
        for i, score in enumerate(relevance_scores[:k]):
            if score > 0:
                return 1.0 / (i + 1)
        return 0.0

    def __call__(self, model):
        print(f"Starting evaluation on {len(self.dev_data)} queries...")

        ndcg_scores = {k: [] for k in self.k_values}
        mrr_scores = {k: [] for k in self.k_values}

        for entry in tqdm(self.dev_data, desc="Evaluating"):
            try:
                query = entry['query']
                candidates = entry['candidates']
                cand_items = list(candidates.values())

                if len(cand_items) < 2:
                    continue

                # Item 0 is Positive, Items 1+ are Negatives
                pos_doc = cand_items[0]
                neg_docs = cand_items[1:]

                # Combine into a single list for the model
                all_docs = [pos_doc] + neg_docs

                # Create Ground Truth labels: 1 for first doc, 0 for the rest
                labels = [1] + [0] * len(neg_docs)

                # Create pairs [[query, doc1], [query, doc2], ...]
                pairs = [[query, doc] for doc in all_docs]

                # Predict
                pred_scores = model.predict(pairs)

                # Sort results by score (descending)
                ranked_results = sorted(zip(labels, pred_scores), key=lambda x: x[1], reverse=True)
                ranked_labels = [x[0] for x in ranked_results]

                # Calculate metrics
                for k in self.k_values:
                    ndcg_scores[k].append(self.compute_ndcg_at_k(ranked_labels, k))
                    mrr_scores[k].append(self.compute_mrr_at_k(ranked_labels, k))

            except Exception as e:
                # print(f"Error processing entry: {e}") 
                continue

        print("\n" + "="*40)
        print(f"FINAL EVALUATION RESULTS")
        print("="*40)
        for k in self.k_values:
            avg_ndcg = np.mean(ndcg_scores[k])
            avg_mrr = np.mean(mrr_scores[k])
            print(f"NDCG@{k:<2}: {avg_ndcg:.4f} | MRR@{k:<2}: {avg_mrr:.4f}")
        print("="*40 + "\n")
