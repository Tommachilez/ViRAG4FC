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

def sanitize_text(text):
    """Standardizes text for robust matching (removes newlines/tabs/extra spaces)."""
    if not text:
        return ""
    return " ".join(text.split()).strip()

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

        # if not os.path.exists(model_path):
        #     print(f"Error: Model path '{model_path}' does not exist.")
        #     sys.exit(1)

        try:
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

    def score_maxp(self, query, doc_text, window_size=250, stride=100):
        """
        Splits long document into overlapping windows, scores each, 
        and returns the MAXIMUM score (MaxP).
        """
        tokens = doc_text.split()

        # Default fallback values
        default_score = -9999.0 if not self.use_sigmoid else 0.0

        if not tokens:
            return default_score, ""

        if len(tokens) <= window_size:
            windows = [doc_text]
        else:
            windows = []
            for i in range(0, len(tokens), stride):
                chunk = " ".join(tokens[i : i + window_size])
                windows.append(chunk)
                if i + window_size >= len(tokens):
                    break

        pairs = [[query, w] for w in windows]
        scores = self.score_batch(pairs)

        if not scores:
            return default_score, "", -1

        # Find the index of the highest score
        max_val = max(scores)
        max_idx = scores.index(max_val)
        best_chunk = windows[max_idx]

        return max_val, best_chunk, max_idx


def load_doc_mapping(csv_path: str, id_col: str, doc_col: str) -> dict:
    """Loads Canonical ID -> Text mapping."""
    docs = {}
    print(f"Loading document mapping from {csv_path}...")
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rid = row.get(id_col)
                txt = row.get(doc_col)
                if rid:
                    docs[str(rid).strip()] = txt if txt else ""
    except Exception as e:
        print(f"Error reading doc mapping: {e}")
        sys.exit(1)
    return docs

def load_query_mapping(csv_path: str, id_col: str, query_col: str) -> dict:
    """Loads Text -> Canonical ID mapping for queries."""
    # We need reverse lookup (Text -> ID) because mining output only has text
    mapping = {}
    print(f"Loading query mapping from {csv_path}...")
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rid = row.get(id_col)
                txt = row.get(query_col)
                if rid and txt:
                    # Sanitize key for robust lookup
                    clean_txt = sanitize_text(txt)
                    mapping[clean_txt] = str(rid).strip()
    except Exception as e:
        print(f"Error reading query mapping: {e}")
        sys.exit(1)

    print(f"Loaded {len(mapping)} queries into mapping.")
    return mapping


def count_lines(filepath):
    """Counts lines in a file for tqdm total."""
    print(f"Counting lines in {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f)


def atomic_save_pickle(data, filepath):
    """
    Saves pickle atomically. Writes to .tmp first, then renames.
    Prevents corruption if script crashes during write.
    """
    temp_path = filepath + ".tmp"
    with gzip.open(temp_path, 'wb') as f:
        pickle.dump(data, f)
    os.replace(temp_path, filepath) # Atomic move


def main():
    parser = argparse.ArgumentParser(description="Calculate ViRanker scores for Distillation.")

    # Required Arguments
    parser.add_argument("--doc_mapping", required=True, help="CSV with columns: doc_id, document")
    parser.add_argument("--query_mapping", required=True, help="CSV with columns: query_id, query")
    parser.add_argument("--mining_jsonl", required=True, help="Path to JSONL from pyserini_mining.")
    parser.add_argument("--output_dir", required=True, help="Directory to save output files.")
    parser.add_argument("--model_path", default='namdp-ptit/ViRanker', help="Path to the local ViRanker checkpoint directory.")

    # Optional Arguments
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--use_sigmoid", action="store_true", help="If set, apply sigmoid to squash logits to [0,1].")
    parser.add_argument("--maxp", action="store_true", help="Enable MaxP scoring (250w/100s). Default is FirstP.")
    parser.add_argument("--append", action="store_true", help="Resume processing by loading existing output file.")
    parser.add_argument("--save_every", type=int, default=100, help="Save checkpoint every N queries.")

    # Column Names
    parser.add_argument("--map_doc_id", default="doc_id")
    parser.add_argument("--map_doc_col", default="document")
    parser.add_argument("--map_query_id", default="query_id")
    parser.add_argument("--map_query_col", default="query")

    args = parser.parse_args()

    # 1. Setup Output Directory
    os.makedirs(args.output_dir, exist_ok=True)
    score_file_path = os.path.join(args.output_dir, "scores.pkl.gz")
    maxp_csv_path = os.path.join(args.output_dir, "best_passage.csv")

    # 2. Load Resources
    scorer = ViRankerScorer(args.model_path, batch_size=args.batch_size, use_sigmoid=args.use_sigmoid)

    # Load Mappings
    doc_map = load_doc_mapping(args.doc_mapping, args.map_doc_id, args.map_doc_col)
    query_text_to_id = load_query_mapping(args.query_mapping, args.map_query_id, args.map_query_col)

    # 3. Handle Resume/Append Logic
    full_scores = {}
    if args.append and os.path.exists(score_file_path):
        print(f"Append mode: Loading scores from {score_file_path}...")
        try:
            with gzip.open(score_file_path, 'rb') as f:
                full_scores = pickle.load(f)
            print(f"Found {len(full_scores)} existing queries in history.")
        except Exception as e:
            print(f"Warning: Pickle corrupted ({e}). Starting fresh.")
            full_scores = {}

    # Initialize CSV variables
    maxp_csv_file = None
    csv_writer = None

    # ONLY initialize the CSV file if maxp is enabled
    if args.maxp:
        file_mode = 'a' if (args.append and os.path.exists(maxp_csv_path)) else 'w'
        maxp_csv_file = open(maxp_csv_path, file_mode, newline='', encoding='utf-8')
        csv_writer = csv.writer(maxp_csv_file)

        # Write header only if we are in 'write' mode (new file)
        if file_mode == 'w':
            csv_writer.writerow(["query_id", "passage_id", "score", "passage_text"])

        print(f"MaxP Enabled: Best passages will be saved to {maxp_csv_path}")

    total_lines = count_lines(args.mining_jsonl)
    print(f"Starting scoring loop. Output: {score_file_path}")

    # Open the mapping file
    with open(args.mining_jsonl, 'r', encoding='utf-8') as f_in:

        pbar = tqdm(total=total_lines, desc="Scoring", unit="queries")
        processed_count = 0
        skipped_map_count = 0

        for line in f_in:
            try:
                entry = json.loads(line)
            except:
                pbar.update(1)
                continue

            query_text = entry.get('query', '')
            candidates = entry.get('candidates', {})

            if not query_text or not candidates:
                pbar.update(1)
                continue

            # 1. Resolve Query ID
            clean_q = sanitize_text(query_text)
            query_id = query_text_to_id.get(clean_q)

            if not query_id:
                # Fallback: Try exact match without sanitization if sanitized failed
                query_id = query_text_to_id.get(query_text)

            if not query_id:
                # If we can't identify the query ID from the mapping, we can't use this row
                skipped_map_count += 1
                pbar.update(1)
                continue

            # 2. Check if already processed
            if query_id in full_scores:
                pbar.update(1)
                continue

            # 3. Score Candidates
            q_scores = {}
            for doc_id, val in candidates.items():
                doc_id_str = str(doc_id).strip()

                # Get text source
                text_to_score = ""

                # If 'val' looks like a document string, use it
                if isinstance(val, str) and len(val) > 5:
                    text_to_score = val
                else:
                    # Otherwise look up in doc_map
                    text_to_score = doc_map.get(doc_id_str, "")

                if not text_to_score:
                    continue

                # Run Inference
                if args.maxp:
                    # Capture BOTH score and best text chunk
                    score, best_chunk, best_idx = scorer.score_maxp(query_text, text_to_score)

                    if best_idx != -1:
                        # KEY CHANGE: Construct the ID as "doc_id#index"
                        # This MUST match the output of create_training_triples_with_maxp.py
                        passage_id = f"{doc_id_str}#{best_idx}"

                        # Save to scores dict with the specific PASSAGE ID
                        q_scores[passage_id] = float(score)

                        # Log to CSV
                        csv_writer.writerow([query_id, passage_id, float(score), best_chunk])
                else:
                    res = scorer.score_batch([[query_text, text_to_score]])
                    score = res[0]
                    q_scores[doc_id_str] = float(score)

            # 4. Save
            if q_scores:
                full_scores[query_id] = q_scores
                processed_count += 1

                # Checkpoint
                if processed_count % args.save_every == 0:
                    atomic_save_pickle(full_scores, score_file_path)
                    if maxp_csv_file:
                        maxp_csv_file.flush() # Ensure CSV data is written to disk

            pbar.update(1)

        pbar.close()

    # Final Save
    print(f"Saving final scores for {len(full_scores)} queries...")
    atomic_save_pickle(full_scores, score_file_path)

    if maxp_csv_file:
        maxp_csv_file.close()

    print(f"Total Queries Scored: {processed_count}")
    print(f"Skipped (Query not in mapping): {skipped_map_count}")

if __name__ == "__main__":
    main()
