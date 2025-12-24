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

        if not tokens:
            return -9999.0 if not self.use_sigmoid else 0.0

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


def clean_text(text):
    """Removes tabs and newlines for safe TSV writing."""
    return text.replace('\t', ' ').replace('\n', ' ').strip()


def ensure_mapping_consistency(mapping_path, jsonl_path, target_count):
    """
    Ensures the TSV mapping file exists and has exactly 'target_count' lines
    corresponding to the pickle. Rebuilds it from JSONL if missing/short.
    """
    # 1. Check current state of mapping file
    current_lines = []
    if os.path.exists(mapping_path):
        with open(mapping_path, 'r', encoding='utf-8') as f:
            current_lines = f.readlines()

    current_count = len(current_lines)

    # CASE A: Perfect Sync
    if current_count == target_count:
        print(f"Mapping file check: OK ({target_count} lines).")
        return

    # CASE B: Mapping is too long (ghost entries) -> Truncate
    if current_count > target_count:
        print(f"Mapping file check: Too long ({current_count} > {target_count}). Truncating...")
        with open(mapping_path, 'w', encoding='utf-8') as f:
            f.writelines(current_lines[:target_count])
        return

    # CASE C: Mapping is missing or too short -> Reconstruct
    # This happens if TSV was deleted or crashed before flushing
    print(f"Mapping file check: Incomplete or Missing ({current_count} < {target_count}). RECONSTRUCTING...")

    processed_so_far = 0
    reconstructed_lines = []

    # We scan the input JSONL just to rebuild the mapping
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=target_count, desc="Rebuilding Mapping", unit="lines"):
            if processed_so_far >= target_count:
                break

            try:
                entry = json.loads(line)
                query_text = entry.get('query', '')
                candidates = entry.get('candidates', {})

                if not query_text or not candidates: continue

                # Format: ID \t Text
                clean_q = clean_text(query_text)
                reconstructed_lines.append(f"{processed_so_far}\t{clean_q}\n")
                processed_so_far += 1
            except:
                continue

    # Write the rebuilt file from scratch
    with open(mapping_path, 'w', encoding='utf-8') as f:
        f.writelines(reconstructed_lines)

    print(f"Mapping file reconstructed successfully with {len(reconstructed_lines)} lines.")


def main():
    parser = argparse.ArgumentParser(description="Calculate ViRanker scores for Distillation.")

    # Required Arguments
    parser.add_argument("--csv", required=True, help="Path to CSV file containing documents.")
    parser.add_argument("--mining_jsonl", required=True, help="Path to JSONL from pyserini_mining (contains candidates).")
    parser.add_argument("--output_dir", required=True, help="Directory to save output files.")
    parser.add_argument("--model_path", default='namdp-ptit/ViRanker', help="Path to the local ViRanker checkpoint directory.")

    # Optional Arguments
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--doc_col", type=str, default="document", help="Column name for document text in CSV")
    parser.add_argument("--use_sigmoid", action="store_true", help="If set, apply sigmoid to squash logits to [0,1].")
    parser.add_argument("--maxp", action="store_true", help="Enable MaxP scoring (250w/100s). Default is FirstP.")
    parser.add_argument("--append", action="store_true", help="Resume processing by loading existing output file.")
    parser.add_argument("--save_every", type=int, default=100, help="Save checkpoint every N queries.")

    args = parser.parse_args()

    # 1. Setup Output Directory
    os.makedirs(args.output_dir, exist_ok=True)
    score_file_path = os.path.join(args.output_dir, "scores.pkl.gz")
    mapping_file_path = os.path.join(args.output_dir, "query_mapping.tsv")

    # 2. Load Resources
    scorer = ViRankerScorer(
        model_path=args.model_path,
        batch_size=args.batch_size,
        use_sigmoid=args.use_sigmoid
    )
    documents = load_documents(args.csv, args.doc_col)

    # 3. Handle Resume/Append Logic
    full_scores = {}
    already_processed_count = 0

    if args.append and os.path.exists(score_file_path):
        print(f"Append mode: Loading scores from {score_file_path}...")
        try:
            with gzip.open(score_file_path, 'rb') as f:
                full_scores = pickle.load(f)
            already_processed_count = len(full_scores)
            print(f"Found {already_processed_count} existing scores.")

            ensure_mapping_consistency(mapping_file_path, args.mining_jsonl, already_processed_count)

        except Exception as e:
            print(f"Warning: Pickle corrupted ({e}). Starting fresh.")
            full_scores = {}
            already_processed_count = 0
    else:
        # If not appending or pickle missing, ensure we start fresh for TSV too
        if os.path.exists(mapping_file_path):
            print("Starting fresh: Overwriting existing mapping file.")
            open(mapping_file_path, 'w').close() # Create empty file immediately

    query_counter = already_processed_count
    total_lines = count_lines(args.mining_jsonl)

    print(f"Starting scoring loop. Output: {args.output_dir}")

    # Open the mapping file
    # buffering=1 means line-buffered (good for text files)
    with open(args.mining_jsonl, 'r', encoding='utf-8') as f_in, \
         open(mapping_file_path, 'a', encoding='utf-8', buffering=1) as f_map:

        pbar = tqdm(total=total_lines, desc="Processing", unit="queries")
        valid_inputs_seen = 0

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

            # Skip already processed
            if valid_inputs_seen < already_processed_count:
                valid_inputs_seen += 1
                pbar.update(1)
                continue

            # --- Scoring ---
            q_scores = {}
            for doc_id, val in candidates.items():
                # Resolve text
                if isinstance(val, str) and len(val) > 10:
                    text_to_score = val
                else:
                    text_to_score = documents.get(str(doc_id))

                if not text_to_score:
                    continue

                if args.maxp:
                    score = scorer.score_maxp(query_text, text_to_score)
                else:
                    res = scorer.score_batch([[query_text, text_to_score]])
                    score = res[0]

                q_scores[str(doc_id)] = float(score)

            # Save Results
            if q_scores:
                full_scores[query_counter] = q_scores

                # Write to TSV immediately
                clean_q = clean_text(query_text)
                f_map.write(f"{query_counter}\t{clean_q}\n")

                query_counter += 1
                valid_inputs_seen += 1

                # Checkpoint the pickle
                if valid_inputs_seen % args.save_every == 0:
                    atomic_save_pickle(full_scores, score_file_path)
                    f_map.flush() # Ensure TSV is safe on disk too

            pbar.update(1)

        pbar.close()

    # Final Save
    print(f"Saving final scores for {len(full_scores)} queries...")
    atomic_save_pickle(full_scores, score_file_path)
    print("Done.")

if __name__ == "__main__":
    main()
