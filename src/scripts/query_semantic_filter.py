import sys
import json
import csv
import argparse
import torch
import math
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def sigmoid(x):
    """Normalizes a raw logit score to [0, 1]."""
    return 1 / (1 + math.exp(-x))

class ViRankerScorer:
    def __init__(self, model_name='namdp-ptit/ViRanker', device=None, batch_size=16):
        self.batch_size = batch_size

        # Determine device
        if device:
            self.device = device
        else:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"

        print(f"Loading ViRanker ({model_name}) on {self.device}...")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)

    def score_batch(self, pairs):
        """
        Runs inference on a batch of [query, document] pairs.
        Returns a list of float scores normalized by sigmoid.
        """
        if not pairs:
            return []

        # Tokenize
        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=512
        ).to(self.device)

        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs, return_dict=True)
            # Flatten logits
            logits = outputs.logits.view(-1).float().cpu().tolist()

        # Normalize
        scores = [sigmoid(l) for l in logits]
        return scores

def load_documents(csv_path: str, content_col: str) -> dict:
    docs = {}
    print(f"Loading documents from {csv_path} (using col: '{content_col}')...")
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            if content_col not in reader.fieldnames:
                print(f"Error: Column '{content_col}' not found in CSV. Available: {reader.fieldnames}")
                sys.exit(1)

            for row in reader:
                if 'id' in row:
                    doc_id = str(row['id']).strip()
                    docs[doc_id] = row[content_col]
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)

    print(f"Loaded {len(docs)} documents.")
    return docs

def get_processed_ids(output_dir: Path, filenames: list) -> set:
    """Scans output files to find IDs that are already processed."""
    processed_ids = set()
    for fname in filenames:
        fpath = output_dir / fname
        if fpath.exists():
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            entry = json.loads(line)
                            if 'id' in entry:
                                processed_ids.add(str(entry['id']).strip())
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                print(f"Warning: Could not read {fpath}: {e}")
    return processed_ids

def main():
    parser = argparse.ArgumentParser(description="Calculate ViRanker scores.")
    parser.add_argument("--csv", required=True, help="Path to CSV file containing documents.")
    parser.add_argument("--jsonl", required=True, help="Path to JSONL file containing queries.")
    parser.add_argument("--column", required=True, help="The CSV column to score against.")
    parser.add_argument("--output_dir", required=True, help="Directory to save output JSONL files.")
    parser.add_argument("--quota", type=int, default=None, help="Total number of documents to process.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for inference.")
    parser.add_argument("--append", action="store_true", help="Resume processing by appending to existing files.")

    args = parser.parse_args()

    # 1. Setup Files & Resume Logic
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    output_filenames = ["viranker_keyword.jsonl", "viranker_natural.jsonl", "viranker_semantic.jsonl"]
    processed_ids = set()
    file_mode = "w"

    if args.append:
        print("Checking for existing progress...")
        processed_ids = get_processed_ids(out_dir, output_filenames)
        if processed_ids:
            print(f"Found {len(processed_ids)} previously processed documents. Resuming...")
            file_mode = "a"
        else:
            print("No existing data found. Starting fresh.")

    # 2. Quota Calculation
    # If quota is 1000 and we have processed 200, we only need to process 800 more.
    session_quota = None
    if args.quota is not None:
        if args.quota <= len(processed_ids):
            print(f"Quota ({args.quota}) already met by existing data ({len(processed_ids)}). Exiting.")
            sys.exit(0)
        session_quota = args.quota - len(processed_ids)
        print(f"Target quota: {args.quota}. Remaining to process: {session_quota}")

    # 3. Load Resources
    scorer = ViRankerScorer(batch_size=args.batch_size)
    documents = load_documents(args.csv, args.column)

    # 4. Open Output Files
    handles = {
        "keyword": open(out_dir / "viranker_keyword.jsonl", file_mode, encoding="utf-8"),
        "natural": open(out_dir / "viranker_natural.jsonl", file_mode, encoding="utf-8"),
        "semantic": open(out_dir / "viranker_semantic.jsonl", file_mode, encoding="utf-8"),
    }

    # 5. Processing Loop
    batch_pairs = []
    batch_meta = []

    # We estimate total for tqdm (either session_quota or total lines in file if unknown)
    total_steps = session_quota if session_quota else None

    count_processed_session = 0

    print(f"Starting processing loop (Mode: {file_mode})...")

    try:
        with open(args.jsonl, 'r', encoding='utf-8') as f:
            # If we don't have a quota, we might want to count file lines for tqdm,
            # but reading a huge file just for that is slow. We'll use tqdm with dynamic updating.
            pbar = tqdm(total=total_steps, desc="Scoring", unit="docs")

            for line in f:
                # Quota Check
                if session_quota is not None and count_processed_session >= session_quota:
                    break

                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                doc_id = str(entry.get('id')).strip()

                # --- SKIP LOGIC ---
                if doc_id in processed_ids:
                    continue

                queries = entry.get('generated_queries', [])

                if doc_id not in documents:
                    continue

                doc_text = documents[doc_id]

                # Collect pairs for this document
                for q in queries:
                    q_text = q.get('query')
                    q_type = q.get('type', '').lower()

                    if not q_text:
                        continue

                    batch_pairs.append([q_text, doc_text])
                    batch_meta.append({
                        "doc_id": doc_id,
                        "query_obj": q,
                        "type": q_type
                    })

                    # Run Batch
                    if len(batch_pairs) >= args.batch_size:
                        scores = scorer.score_batch(batch_pairs)

                        for score, meta in zip(scores, batch_meta):
                            q_type_key = meta['type']
                            target_file = None
                            if "keyword" in q_type_key:
                                target_file = handles["keyword"]
                            elif "natural" in q_type_key:
                                target_file = handles["natural"]
                            elif "semantic" in q_type_key:
                                target_file = handles["semantic"]

                            if target_file:
                                out_obj = {
                                    "id": meta['doc_id'],
                                    "query": meta['query_obj'],
                                    "viranker_score": round(score, 5)
                                }
                                json.dump(out_obj, target_file, ensure_ascii=False)
                                target_file.write("\n")

                        batch_pairs = []
                        batch_meta = []

                # Update Counters
                count_processed_session += 1
                pbar.update(1)

            # Process remaining buffer
            if batch_pairs:
                scores = scorer.score_batch(batch_pairs)
                for score, meta in zip(scores, batch_meta):
                    q_type_key = meta['type']
                    target_file = None
                    if "keyword" in q_type_key:
                        target_file = handles["keyword"]
                    elif "natural" in q_type_key:
                        target_file = handles["natural"]
                    elif "semantic" in q_type_key:
                        target_file = handles["semantic"]

                    if target_file:
                        out_obj = {
                            "id": meta['doc_id'],
                            "query": meta['query_obj'],
                            "viranker_score": round(score, 5)
                        }
                        json.dump(out_obj, target_file, ensure_ascii=False)
                        target_file.write("\n")
            
            pbar.close()

    except Exception as e:
        print(f"Error processing JSONL: {e}")
    finally:
        for h in handles.values():
            h.close()
        print(f"\nDone. Processed {count_processed_session} new documents in this session.")
        print(f"Total processed including previous runs: {count_processed_session + len(processed_ids)}")

if __name__ == "__main__":
    main()
