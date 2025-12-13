import sys
import json
import csv
import argparse
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


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
        Returns a list of float scores.
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
            # Flatten logits to 1D array
            scores = outputs.logits.view(-1).float().cpu().tolist()

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
                    # Normalize ID to string to ensure matching works
                    doc_id = str(row['id']).strip()
                    docs[doc_id] = row[content_col]
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)

    print(f"Loaded {len(docs)} documents.")
    return docs

def main():
    parser = argparse.ArgumentParser(description="Calculate relevance scores using ViRanker.")
    parser.add_argument("--csv", required=True, help="Path to CSV file containing documents.")
    parser.add_argument("--jsonl", required=True, help="Path to JSONL file containing queries.")
    parser.add_argument("--column", required=True, help="The CSV column name to score against (e.g., 'document', 'evidence').")
    parser.add_argument("--output_dir", required=True, help="Directory to save output JSONL files.")
    parser.add_argument("--quota", type=int, default=None, help="Stop after processing N documents (lines from JSONL).")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for inference (reduce if OOM).")

    args = parser.parse_args()

    # 1. Setup
    scorer = ViRankerScorer(batch_size=args.batch_size)
    documents = load_documents(args.csv, args.column)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Output file handles
    handles = {
        "keyword": open(out_dir / "viranker_keyword.jsonl", "w", encoding="utf-8"),
        "natural": open(out_dir / "viranker_natural.jsonl", "w", encoding="utf-8"),
        "semantic": open(out_dir / "viranker_semantic.jsonl", "w", encoding="utf-8"),
    }

    # Buffers for batch processing
    batch_pairs = [] # List of [query, doc]
    batch_meta = []  # List of metadata dicts to reconstruct output later

    count_processed_docs = 0

    print("Processing queries...")

    # We use a progress bar based on lines if possible, or simple iteration
    try:
        with open(args.jsonl, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Reading JSONL"):
                # Quota Check
                if args.quota is not None and count_processed_docs >= args.quota:
                    break

                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                doc_id = str(entry.get('id')).strip()
                queries = entry.get('generated_queries', [])

                # Check if document exists
                if doc_id not in documents:
                    continue

                doc_text = documents[doc_id]

                # Collect pairs for this document
                for q in queries:
                    # Normalize query text key
                    q_text = q.get('query')
                    q_type = q.get('type', '').lower()

                    if not q_text:
                        continue

                    # Add to batch
                    batch_pairs.append([q_text, doc_text])
                    batch_meta.append({
                        "doc_id": doc_id,
                        "query_obj": q,
                        "type": q_type
                    })

                    # PROCESS BATCH IF FULL
                    if len(batch_pairs) >= args.batch_size:
                        scores = scorer.score_batch(batch_pairs)

                        # Write results
                        for score, meta in zip(scores, batch_meta):
                            q_type_key = meta['type']

                            # Determine output file
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

                        # Clear buffers
                        batch_pairs = []
                        batch_meta = []

                count_processed_docs += 1

            # PROCESS REMAINING ITEMS IN BUFFER
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

    except Exception as e:
        print(f"Error processing JSONL: {e}")
    finally:
        for h in handles.values():
            h.close()
        print(f"\nDone. Processed {count_processed_docs} documents.")
        print(f"Results saved to {out_dir}")

if __name__ == "__main__":
    main()
