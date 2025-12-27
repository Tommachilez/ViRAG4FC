import sys
import csv
import argparse
import os
import string
from tqdm import tqdm
from typing import Set

# NLP Imports
try:
    import py_vncorenlp
    from underthesea import text_normalize
except ImportError:
    print("Error: NLP libraries missing. Run: pip install py_vncorenlp underthesea", file=sys.stderr)
    sys.exit(1)

# ==========================================
# CONFIGURATION & PROCESSOR
# ==========================================
STOPWORD_WHITELIST = {
    "không", "chẳng", "chả", "chưa", "phi", "vô", "tránh", "đừng", "chớ",
    "và", "hoặc", "nhưng", "tuy", "nếu", "thì", "vì", "do", "bởi", "tại", "nên", 
    "rằng", "là", "của", "thuộc",
    "tại", "ở", "trong", "ngoài", "trên", "dưới", "giữa", "với", "về", "đến",
    "ai", "gì", "nào", "đâu", "khi", "mấy", "bao_nhiêu", "thế_nào", "sao",
    "bị", "được", "do", "bởi"
}

class VietnameseProcessor:
    def __init__(self, vncorenlp_path: str, stopwords_path: str, use_whitelist: bool = False):
        if not os.path.exists(vncorenlp_path):
            raise FileNotFoundError(f"VnCoreNLP not found at {vncorenlp_path}")

        try:
            self.rdrsegmenter = py_vncorenlp.VnCoreNLP(
                save_dir=vncorenlp_path,
                annotators=["wseg"]
            )
        except Exception as e:
            print(f"Error init VnCoreNLP: {e}")
            sys.exit(1)

        self.use_whitelist = use_whitelist
        self.stopwords = self._load_stopwords(stopwords_path)
        self.punctuation = set(string.punctuation)

    def _load_stopwords(self, path: str) -> Set[str]:
        sw = set()
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    w = line.strip().lower()
                    if w:
                        token = w.replace(' ', '_').replace('-', '_')
                        if self.use_whitelist and token in STOPWORD_WHITELIST:
                            continue
                        sw.add(token)
        return sw

    def process(self, text: str) -> str:
        if not text: return ""
        try:
            text = text_normalize(text.lower())
        except:
            text = text.lower()
        try:
            sents = self.rdrsegmenter.word_segment(text)
            tokens = [t for sent in sents for t in sent.split()]
            valid = [t for t in tokens if t not in self.punctuation and t not in self.stopwords]
            return " ".join(valid)
        except:
            return ""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_tsv", required=True, help="Path to raw test_queries.tsv")
    parser.add_argument("--output_tsv", required=True, help="Path to save segmented queries")
    parser.add_argument("--vncorenlp_path", required=True)
    parser.add_argument("--stopwords_path", required=True)
    args = parser.parse_args()

    print(">>> Initializing NLP Processor...")
    processor = VietnameseProcessor(args.vncorenlp_path, args.stopwords_path, use_whitelist=True)

    print(f">>> Processing {args.input_tsv}...")
    with open(args.input_tsv, 'r', encoding='utf-8') as f_in, \
         open(args.output_tsv, 'w', encoding='utf-8', newline='') as f_out:

        reader = csv.reader(f_in, delimiter='\t')
        writer = csv.writer(f_out, delimiter='\t')

        count = 0
        for row in tqdm(reader):
            if len(row) >= 2:
                q_id = row[0]
                q_raw = row[1]
                q_seg = processor.process(q_raw)

                if q_seg:
                    writer.writerow([q_id, q_seg])
                    count += 1

    print(f">>> Done. {count} queries segmented and saved to {args.output_tsv}")

if __name__ == "__main__":
    main()
