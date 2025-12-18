import json
import logging
import math
import numpy as np
import os
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

    logging.info(f"Loading raw data from {file_path}...")
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

def prepare_training_samples(raw_data):
    """Convert raw JSON objects into InputExample for SentenceTransformers"""
    samples = []
    for data in raw_data:
        query = data.get("query", "")
        # Create pairs
        for pos in data.get("pos", []):
            samples.append(InputExample(texts=[query, pos], label=1.0))
        for neg in data.get("neg", []):
            samples.append(InputExample(texts=[query, neg], label=0.0))
    return samples

def train_viranker():
    # 1. Load Data
    train_raw, dev_raw = load_and_split_data(TRAIN_FILE)
    train_samples = prepare_training_samples(train_raw)

    # 2. Initialize Model
    logging.info(f"Initializing CrossEncoder: {MODEL_NAME}")
    model = CrossEncoder(MODEL_NAME, num_labels=1, max_length=MAX_SEQ_LENGTH)

    # 3. DataLoader
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=BATCH_SIZE)

    # 4. Initialize Evaluator
    evaluator = RankerEvaluator(dev_raw, k_values=[3, 5, 10])

    # 5. Training Loop
    warmup_steps = math.ceil(len(train_dataloader) * NUM_EPOCHS * 0.1)

    logging.info("Starting training...")

    # SentenceTransformers 'fit' does not natively support our custom complex evaluator
    # in the way we want (printing a table).
    # We will pass a simple callback or just run evaluation at the end.
    # To keep it robust, we run training, then run evaluation manually.

    model.fit(
        train_dataloader=train_dataloader,
        epochs=NUM_EPOCHS,
        warmup_steps=warmup_steps,
        output_path=OUTPUT_DIR,
        optimizer_params={'lr': LEARNING_RATE},
        show_progress_bar=True
    )

    logging.info("Training finished. Running final evaluation...")

    # Run Final Evaluation on the Dev Set
    evaluator(model, epoch_idx=NUM_EPOCHS)

    logging.info(f"Model saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    train_viranker()
