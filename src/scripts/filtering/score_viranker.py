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


def sync_mapping_file(mapping_path, target_count):
    """
    Ensures the mapping file has exactly 'target_count' lines.
    Removes 'ghost' entries from a previous crash/interrupted run.
    """
    if not os.path.exists(mapping_path):
        return

    # Check line count first to avoid reading file if unnecessary
    with open(mapping_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    current_len = len(lines)

    if current_len == target_count:
        print(f"Mapping file is in sync ({current_len} lines).")
        return

    if current_len > target_count:
        print(f"Syncing mapping file: Truncating from {current_len} to {target_count} lines...")
        with open(mapping_path, 'w', encoding='utf-8') as f:
            f.writelines(lines[:target_count])
    else:
        # This is the "Warning" case you had.
        # It usually means the user deleted the mapping file or it got corrupted.
        print(f"Warning: Mapping file ({current_len}) is shorter than scores ({target_count}). Mappings may be missing.")


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

    # Mode for opening the mapping file
    mapping_mode = 'w'

    if args.append and os.path.exists(score_file_path):
        print(f"Append mode: Loading existing scores from {score_file_path}...")
        try:
            with gzip.open(score_file_path, 'rb') as f:
                full_scores = pickle.load(f)
            already_processed_count = len(full_scores)
            print(f"Found {already_processed_count} existing queries. Resuming...")
            sync_mapping_file(mapping_file_path, already_processed_count)
            mapping_mode = 'a' # Append to existing mapping
        except Exception as e:
            print(f"Warning: Could not load existing pickle ({e}). Starting fresh.")
            full_scores = {}
            already_processed_count = 0
            mapping_mode = 'w'

    query_counter = already_processed_count
    total_lines = count_lines(args.mining_jsonl)

    print(f"Starting scoring loop. Output: {args.output_dir}")

    # Open the mapping file
    # buffering=1 means line-buffered (good for text files)
    with open(args.mining_jsonl, 'r', encoding='utf-8') as f_in, \
         open(mapping_file_path, mapping_mode, encoding='utf-8', buffering=1) as f_map:

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
                    # UPDATED: Using safe default parameters (200/100) inside the method
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
