import json
import logging
import os
import argparse
from sklearn.model_selection import train_test_split
from datasets import Dataset

from sentence_transformers import CrossEncoder
from sentence_transformers.cross_encoder.trainer import CrossEncoderTrainer
from sentence_transformers.cross_encoder.training_args import CrossEncoderTrainingArguments

from .evaluator import RankerEvaluator

# --- Configuration Constants ---
# MODEL_NAME = "BAAI/bge-m3"
MODEL_NAME = "namdp-ptit/ViRanker"
TRAIN_FILE = "train_data.jsonl"
OUTPUT_DIR = "./viranker_checkpoint"
BATCH_SIZE = 4
NUM_EPOCHS = 1
LEARNING_RATE = 1e-5
MAX_SEQ_LENGTH = 1024
DEV_SPLIT_RATIO = 0.1
GRADIENT_ACCUMULATION_STEPS = 16

logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO
)

# --- Helper for MaxP ---
def sliding_window(text, window_size=250, stride=100):
    """
    Splits text into overlapping chunks of words.
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

def load_and_split_data(file_path, dev_split_ratio):
    all_data = []
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                all_data.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    train_raw, dev_raw = train_test_split(all_data, test_size=dev_split_ratio, random_state=42)
    logging.info(f"Data Split -> Train: {len(train_raw)} | Dev: {len(dev_raw)}")
    return train_raw, dev_raw

# Validation Helper: Filter out empty/garbage text
def is_valid_text(t):
    # Must be string, not empty, and have at least 5 meaningful characters
    return isinstance(t, str) and len(t.strip()) >= 5

def prepare_dataset(raw_data: list, use_maxp=False):
    """
    Converts raw data into a dictionary format suitable for HF Dataset.
    Columns: 'sentence1' (Query), 'sentence2' (Doc), 'label' (Score)
    """
    queries = []
    docs = []
    labels = []

    logging.info(f"Preparing dataset (MaxP={'Enabled' if use_maxp else 'Disabled'})...") 

    for data in raw_data:
        try:
            query = data['query']
            candidates = data['candidates']
            cand_items = list(candidates.values())

            if len(cand_items) < 2:
                continue

            pos_text = cand_items[0]
            neg_texts = cand_items[1:]

            # Validate Positive Text
            if not is_valid_text(pos_text):
                continue

            # Filter Negatives (Remove garbage negatives)
            valid_neg_texts = [n for n in neg_texts if is_valid_text(n)]
            if not valid_neg_texts:
                continue

            # --- Logic Branching ---
            if use_maxp:
                # MaxP Logic: Split positives into chunks, consider ALL as relevant
                pos_chunks = sliding_window(pos_text)

                # Positive Samples
                for p_chunk in pos_chunks:
                    queries.append(query)
                    docs.append(p_chunk)
                    labels.append(1.0) # Float for CE loss (usually)

                # Negative Samples (Using first window for efficiency)
                for n_text in neg_texts:
                    n_chunk_0 = sliding_window(n_text)[0]
                    queries.append(query)
                    docs.append(n_chunk_0)
                    labels.append(0.0)

            else:
                # Standard FirstP Logic (Naive Truncation)
                for neg_text in neg_texts:
                    # 1. Add Positive
                    queries.append(query)
                    docs.append(pos_text)
                    labels.append(1.0)

                    # 2. Add Negative
                    queries.append(query)
                    docs.append(neg_text)
                    labels.append(0.0)

        except Exception as e:
            continue

    logging.info(f"Generated {len(labels)} training pairs.")

    if len(labels) == 0:
        raise ValueError("Dataset is empty! Check your data or validation logic.")

    # Create HF Dataset
    return Dataset.from_dict({
        "sentence1": queries,
        "sentence2": docs,
        "label": labels
    })

def train_viranker(args):
    # 1. Load Data & Prepare HF Dataset
    train_raw, dev_raw = load_and_split_data(args.train_file, args.dev_split_ratio)
    train_dataset = prepare_dataset(train_raw, use_maxp=args.maxp)

    # 2. Initialize Model
    model_path = args.model_name
    if args.continue_train:
        # Check if output_dir has valid checkpoints
        if os.path.exists(args.output_dir) and any(x.startswith("checkpoint") for x in os.listdir(args.output_dir)):
            logging.info(f"🔄 Checkpoint detected. Trainer will resume from: {args.output_dir}")

    logging.info(f"Initializing CrossEncoder from: {model_path}")

    model = CrossEncoder(model_path, num_labels=1, max_length=args.max_seq_length)

    # 3. Initialize Evaluator
    evaluator = RankerEvaluator(dev_raw, k_values=[3, 5, 10], use_maxp=args.maxp)

    # 4. Training Arguments

    training_args = CrossEncoderTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        # max_grad_norm=1.0,
        warmup_ratio=0.1,
        fp16=True,             # Enable mixed precision for speed
        save_total_limit=5,    # Only keep the last 5 checkpoints to save space
        logging_steps=100,

        # KEY CHANGES FOR SAVING:
        save_strategy="steps" if args.save_every > 0 else "no",
        save_steps=args.save_every if args.save_every > 0 else 0,

        # Evaluation settings
        eval_strategy="no", # We use our custom evaluator at the end, or you can add it to callbacks
    )

    # 5. Initialize Trainer
    trainer = CrossEncoderTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        # We don't pass 'evaluator' here because CrossEncoderTrainer expects a 
        # specific CEBinaryClassificationEvaluator type.
        # We will run our custom RankerEvaluator manually at the end.
    )

    # 6. Train
    logging.info("Starting training...")
    trainer.train(resume_from_checkpoint=args.continue_train)

    # 7. Final Save & Eval
    logging.info("Training finished. Saving final model...")
    trainer.save_model(args.output_dir)

    # Run custom evaluation
    logging.info("Running final evaluation...")
    evaluator(model) # 'model' object is updated in-place by trainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ViRanker Model")
    parser.add_argument("--train_file", type=str, default=TRAIN_FILE)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--model_name", type=str, default=MODEL_NAME)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=GRADIENT_ACCUMULATION_STEPS)
    parser.add_argument("--num_epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--max_seq_length", type=int, default=MAX_SEQ_LENGTH)
    parser.add_argument("--dev_split_ratio", type=float, default=DEV_SPLIT_RATIO)
    parser.add_argument("--continue_train", action="store_true", help="Resume from checkpoint if available")
    parser.add_argument("--maxp", action="store_true", help="Enable MaxP sliding window training.")
    parser.add_argument("--save_every", type=int, default=500, help="Save checkpoint every N steps.")

    train_viranker(parser.parse_args())
