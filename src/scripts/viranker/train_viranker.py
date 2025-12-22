import json
import logging
import math
import numpy as np
import os
import argparse
from torch.utils.data import DataLoader
from sentence_transformers import CrossEncoder, InputExample
from sklearn.model_selection import train_test_split

from .evaluator import RankerEvaluator

# --- Configuration ---
# MODEL_NAME = "BAAI/bge-m3"
MODEL_NAME = "namdp-ptit/ViRanker"
TRAIN_FILE = "train_data.jsonl"
OUTPUT_DIR = "./viranker_checkpoint"
BATCH_SIZE = 4
NUM_EPOCHS = 1
LEARNING_RATE = 2e-5
MAX_SEQ_LENGTH = 512
# Evaluation Split (0.1 = 10% of data used for Dev/Metrics)
DEV_SPLIT_RATIO = 0.1

logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO
)

# --- Main Training Logic ---

def load_and_split_data(file_path):
    all_data = []

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")

    # logging.info(f"Loading raw data from {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                all_data.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    # Split into Train and Dev (Validation) sets
    train_raw, dev_raw = train_test_split(all_data, test_size=DEV_SPLIT_RATIO, random_state=42)
    logging.info(f"Data Split -> Train: {len(train_raw)} | Dev: {len(dev_raw)}")

    return train_raw, dev_raw

def prepare_training_samples(raw_data: list):
    """
    Converts a list of raw data dicts into a list of InputExample objects.
    """
    samples = []

    logging.info(f"Preparing training samples from {len(raw_data)} raw records...") 

    for data in raw_data: # Iterate over the list directly
        try:
            query = data['query']
            candidates = data['candidates']

            # Convert dict items to a list to access by index
            # Python 3.7+ preserves insertion order, so the first item 
            # is the Positive (as inserted by our mining script).
            cand_items = list(candidates.values())

            if len(cand_items) < 2:
                # Need at least 1 pos and 1 neg
                continue

            # First item is Positive
            pos_text = cand_items[0]

            # Remaining items are Negatives
            neg_texts = cand_items[1:]

            # Create a Triplet (Query, Pos, Neg) for every negative found
            for neg_text in neg_texts:
                # 1. Positive Pair
                samples.append(InputExample(texts=[query, pos_text], label=1.0))
                # 2. Negative Pair
                samples.append(InputExample(texts=[query, neg_text], label=0.0))

        except Exception as e:
            print(f"Skipping bad line: {e}")
            continue

    print(f"Loaded {len(samples)} training triples.")
    return samples

def train_viranker(args):
    # 1. Load Data
    train_raw, dev_raw = load_and_split_data(args.train_file)
    train_samples = prepare_training_samples(train_raw)

    # 2. Initialize Model
    logging.info(f"Initializing CrossEncoder: {args.model_name}")
    model = CrossEncoder(args.model_name, num_labels=1, max_length=args.max_seq_length)

    # 3. DataLoader
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=args.batch_size)

    # 4. Initialize Evaluator
    evaluator = RankerEvaluator(dev_raw, k_values=[3, 5, 10])

    # 5. Training Loop
    warmup_steps = math.ceil(len(train_dataloader) * args.num_epochs * 0.1)

    logging.info("Starting training...")

    # SentenceTransformers 'fit' does not natively support our custom complex evaluator
    # in the way we want (printing a table).
    # We will pass a simple callback or just run evaluation at the end.
    # To keep it robust, we run training, then run evaluation manually.

    model.fit(
        train_dataloader=train_dataloader,
        epochs=args.num_epochs,
        warmup_steps=warmup_steps,
        output_path=args.output_dir,
        optimizer_params={'lr': args.learning_rate},
        show_progress_bar=True
    )

    logging.info("Training finished. Running final evaluation...")

    # Run Final Evaluation on the Dev Set
    evaluator(model, epoch_idx=args.num_epochs)

    logging.info(f"Model saved to: {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ViRanker Model")
    parser.add_argument("--train_file", type=str, default=TRAIN_FILE, help="Path to training data file (JSONL format)")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR, help="Directory to save the trained model")
    parser.add_argument("--model_name", type=str, default=MODEL_NAME, help="Pretrained model name or path")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Training batch size")
    parser.add_argument("--num_epochs", type=int, default=NUM_EPOCHS, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE, help="Learning rate for optimizer")
    parser.add_argument("--max_seq_length", type=int, default=MAX_SEQ_LENGTH, help="Maximum sequence length for the model")
    parser.add_argument("--dev_split_ratio", type=float, default=DEV_SPLIT_RATIO, help="Proportion of data to use for development/validation")
    train_viranker(parser.parse_args())
