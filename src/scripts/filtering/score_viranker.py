import sys
import json
import csv
import argparse
import math
import pickle
import gzip
import os

import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

class ViRankerScorer:
    def __init__(self, model_path, device=None, batch_size=16, use_sigmoid=False):
        self.batch_size = batch_size
        self.use_sigmoid = use_sigmoid  # Store the flag

        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading ViRanker from checkpoint: {model_path} on {self.device}...")
        print(f"Output Mode: {'Sigmoid (0-1)' if self.use_sigmoid else 'Raw Logits'}")

        if not os.path.exists(model_path):
            print(f"Error: Model path '{model_path}' does not exist.")
            sys.exit(1)

        try:
            # Load from local checkpoint directory
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
            sys.exit(1)

    def score_batch(self, pairs):
        """Scores a simple list of [query, text] pairs."""
        if not pairs:
            return []

        # Tokenize (Keep max_length=512 for the chunks)
        inputs = self.tokenizer(
            pairs, padding=True, truncation=True,
            return_tensors='pt', max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs, return_dict=True)
            logits = outputs.logits.view(-1).float().cpu().tolist()

        # CONDITIONAL SIGMOID
        if self.use_sigmoid:
            return [sigmoid(l) for l in logits]

        return logits

    def score_maxp(self, query, doc_text, window_size=512, stride=256):
        """
        Splits long document into overlapping windows, scores each, 
        and returns the MAXIMUM score (MaxP).
        """
        # Approximate token split (accurate enough for windowing)
        tokens = doc_text.split()

        if not tokens:
            return -9999.0 if not self.use_sigmoid else 0.0

        if len(tokens) <= window_size:
            windows = [doc_text]
        else:
            windows = []
            for i in range(0, len(tokens), stride):
                # Reconstruct string from token slice
                chunk = " ".join(tokens[i : i + window_size])
                windows.append(chunk)
                # Stop if we've reached the end
                if i + window_size >= len(tokens):
                    break

        # Prepare pairs for batching
        pairs = [[query, w] for w in windows]

        # Score all chunks
        scores = self.score_batch(pairs)

        return max(scores) if scores else (-9999.0 if not self.use_sigmoid else 0.0)


def load_documents(csv_path: str, content_col: str) -> dict:
    docs = {}
    print(f"Loading documents from {csv_path}...")
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'id' in row:
                    docs[str(row['id']).strip()] = row[content_col]
    except Exception as e:
        print(f"Error reading document CSV: {e}")
        sys.exit(1)
    return docs


def main():
    parser = argparse.ArgumentParser(description="Calculate ViRanker scores for Distillation.")

    # Required Arguments
    parser.add_argument("--csv", required=True, help="Path to CSV file containing documents.")
    parser.add_argument("--mining_jsonl", required=True, help="Path to JSONL from pyserini_mining (contains candidates).")
    parser.add_argument("--output_pkl", required=True, help="Path to save the .pkl.gz score file.")
    parser.add_argument("--model_path", required=True, default='namdp-ptit/ViRanker', help="Path to the local ViRanker checkpoint directory (e.g., ./viranker_checkpoint).")

    # Optional Arguments
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--doc_col", type=str, default="document", help="Column name for document text in CSV")
    parser.add_argument("--use_sigmoid", action="store_true", help="If set, apply sigmoid to squash logits to [0,1].")

    args = parser.parse_args()

    # 1. Load Resources
    # Pass the local model path to the Scorer
    scorer = ViRankerScorer(
        model_path=args.model_path,
        batch_size=args.batch_size,
        use_sigmoid=args.use_sigmoid
    )

    # Load raw documents to map IDs from candidates back to text
    documents = load_documents(args.csv, args.doc_col)

    # 2. Data Structures
    # Structure: { query_id_int: { doc_id_str: score_float } }
    full_scores = {}

    # 3. Processing Loop
    print("Starting scoring loop...")

    # We use a simple counter for Query ID if not present, to match DeeperImpact format
    query_counter = 0

    with open(args.mining_jsonl, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Scoring Candidates"):
            try:
                entry = json.loads(line)
            except:
                print("Skipping bad line (not valid JSON).")
                continue

            query_text = entry.get('query', '')
            candidates = entry.get('candidates', {}) # Expecting {doc_id: doc_text} or just {doc_id: ...}

            if not query_text or not candidates:
                continue

            q_scores = {}

            # Iterate through candidates for this query
            for doc_id, val in candidates.items():
                # If mining file has text, use it. If not, look up in 'documents' dict.
                # Assuming 'val' is the text, or we use 'documents[doc_id]'
                if isinstance(val, str) and len(val) > 10:
                    text_to_score = val
                else:
                    text_to_score = documents.get(str(doc_id))

                if not text_to_score:
                    continue

                # MaxP Score
                score = scorer.score_maxp(query_text, text_to_score)
                q_scores[str(doc_id)] = float(score)

            if q_scores:
                full_scores[query_counter] = q_scores
                query_counter += 1

    # 4. Save to Pickle
    print(f"Saving scores for {len(full_scores)} queries to {args.output_pkl}...")
    with gzip.open(args.output_pkl, 'wb') as f:
        pickle.dump(full_scores, f)
    print("Done.")

if __name__ == "__main__":
    main()
