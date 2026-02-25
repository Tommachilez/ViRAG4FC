# ViRAG4FC: Information Retrieval Leveraging LLM & Contrastive Learning for Fact Checking

ViRAG4FC is an advanced, two-stage Information Retrieval (IR) system designed to solve the challenges of data scarcity and semantic ambiguity in Vietnamese fact-checking. By combining the generative power of **Gemini 2.5 Flash** with a **Hybrid Retrieval** strategy (BM25 + DeeperImpact), this system achieves a state-of-the-art **76.62% accuracy** on unified Vietnamese benchmarks.

## 🏗️ System Architecture

Our pipeline is built on a "System-of-Systems" approach, ensuring that the final verification is always grounded in high-quality, retrieved evidence.

### The Workflow

#### Retriever

<p align="center">
<img src="src/others/retriever.png" alt="Retriever" width="500">
</p>

#### Reader

<p align="center">
<img src="src/others/reader.png" alt="Reader" width="500">
</p>

The system operates in three core phases:

1. **LLM-Driven Data Augmentation**: Utilizing the Gemini API to generate multi-view query triples (Keyword, Natural, and Semantic) to enrich the training corpus.
2. **Hybrid Retrieval**: A combination of lexical precision (BM25) and semantic depth (DeeperImpact).
3. **Evidence-Based Verification**: A Reader module (Gemini 2.5 Flash) that performs zero-shot reasoning over the top-K retrieved documents to provide a final verdict.

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- Gemini API Key
- Java 11+ (for Pyserini/BM25)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/ViRAG4FC.git

# Navigate to the project directory
cd ViRAG4FC

# Install dependencies
pip install -r requirements.txt
```

### Running Retrieval

You can run the retrieval pipeline for a single query or batch processing using `document_retrieval.py`:

#### Single Query Example:

```bash
python document_retrieval.py --query "Your query here" --top_k 10
```

#### Batch Processing Example:

```bash
python document_retrieval.py --input_file data/queries.csv --output_file results/retrieved_docs.jsonl --batch_size 16
```

## 📊 Benchmarks

We unified three isolated datasets (ViFactCheck, ViWikiFC, and ViNumFCR) into a comprehensive 34,811-sample benchmark.

| Retriever + Reader | Accuracy | Macro F1 |
|---------|------|------|
| BM25 + Gemini | 76.40% | 0.74 |
| DeeperImpact + Gemini | 69.73% | 0.69 |
| **Hybrid + Gemini (Ours)** | **76.62%** | **0.77** |

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contributions

Contributions are welcome! Feel free to open issues or submit pull requests.

## Contact

For questions or feedback, please contact <michael.nguyenbathong@gmail.com>.
