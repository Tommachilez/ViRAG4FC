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


def sliding_window(text, window_size=200, stride=100):
    """
    Splits text into overlapping chunks. 
    Note: BERT limit is 512 TOKENS, but string splitting is rough. 
    Safe bet is ~300-400 words if splitting by whitespace.
    """
    tokens = text.split()
    if len(tokens) <= window_size:
        return [text]

    windows = []
    for i in range(0, len(tokens), stride):
        chunk = " ".join(tokens[i : i + window_size])
        windows.append(chunk)
        if i + window_size >= len(tokens):
            break
    return windows


def prepare_training_samples(raw_data: list, use_maxp=False):
    samples = []
    logging.info(f"Preparing training samples (MaxP={'Enabled' if use_maxp else 'Disabled'})...")

    for data in raw_data: # Iterate over the list directly
        try:
            query = data['query']
            candidates = data['candidates']
            cand_items = list(candidates.values())

            if len(cand_items) < 2:
                # Need at least 1 pos and 1 neg
                continue

            pos_text = cand_items[0]
            neg_texts = cand_items[1:]

            if use_maxp:

                pos_chunks = sliding_window(pos_text)

                for neg_text in neg_texts:
                    n_chunk = sliding_window(neg_text)[0]
                    samples.append(InputExample(texts=[query, n_chunk], label=0.0))

                for p_chunk in pos_chunks:
                    samples.append(InputExample(texts=[query, p_chunk], label=1.0))

                # for n_chunks in neg_chunks_list:
                #     for n_chunk in n_chunks:
                #         samples.append(InputExample(texts=[query, n_chunk], label=0.0))

            else:
                # Standard BERT-FirstP (Naive Truncation)
                # Just take the text as-is; model will truncate to 512 tokens
                for neg_text in neg_texts:
                    samples.append(InputExample(texts=[query, pos_text], label=1.0))
                    samples.append(InputExample(texts=[query, neg_text], label=0.0))

        except Exception as e:
            print(f"Skipping bad line: {e}")
            continue

    print(f"Loaded {len(samples)} training triples.")
    return samples

def train_viranker(args):
    # 1. Load Data
    train_raw, dev_raw = load_and_split_data(args.train_file)

    # Pass the maxp flag here
    train_samples = prepare_training_samples(train_raw, use_maxp=args.maxp)

    # 2. Initialize Model (Fresh or Continue)
    model_path = args.model_name

    if args.continue_train:
        if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
            logging.info(f"🔄 Continuing training from checkpoint: {args.output_dir}")
            model_path = args.output_dir
        else:
            logging.warning(f"⚠️ Checkpoint directory {args.output_dir} is empty or missing. Starting fresh from {args.model_name}")

    logging.info(f"Initializing CrossEncoder from: {model_path}")
    model = CrossEncoder(model_path, num_labels=1, max_length=args.max_seq_length)

    # 3. DataLoader
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=args.batch_size)

    # 4. Initialize Evaluator
    evaluator = RankerEvaluator(dev_raw, k_values=[3, 5, 10], use_maxp=args.maxp)

    # 5. Training Loop
    warmup_steps = math.ceil(len(train_dataloader) * args.num_epochs * 0.1)

    logging.info("Starting training...")

    # SentenceTransformers 'fit' does not natively support complex resumption logic (like skipping steps),
    # but loading the model weights allows it to continue learning from where it left off.
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
    evaluator(model)

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

    # NEW ARGUMENT
    parser.add_argument("--maxp", action="store_true", help="Enable MaxP sliding window training (200 words/100 stride).")
    parser.add_argument("--continue_train", action="store_true", help="If set, tries to load model from output_dir to resume training")

    train_viranker(parser.parse_args())
