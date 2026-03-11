# 📓 Project Notebooks Guide

This directory contains Jupyter notebooks used for the development, training, and evaluation of the **ViRAG4FC** framework. The notebooks are organized by their stage in the Information Retrieval (IR) and Retrieval-Augmented Generation (RAG) pipeline.

## 📂 Directory Structure

The notebooks are categorized into three main sub-directories, plus root-level demo scripts:

### 1. Data Handling

These notebooks focus on Exploratory Data Analysis (EDA) and the preparation of Vietnamese datasets for training.

- **EDA Notebooks**: (e.g., `eda_vifc.ipynb`, `eda_vifactcheck.ipynb`) Used to inspect dataset distributions, visualize document token counts, and verify data integrity.

- **Training Data Generation**: (e.g., `bm25_generate_training_triples.ipynb`) Uses Gemini to generate multi-view query triples (Keyword, Natural, Semantic) to enrich the training data for the retriever.

- **Expansion Inspection**: `inspect_expansion.ipynb` allows for the analysis of semantic term expansion and weighting.

### 2. Main Pipeline

Contains the core logic for training models and executing the retrieval pipeline.

- **Training**: `retrain_deeperimpact.ipynb` and `train_viranker_firstp.ipynb` are used to fine-tune the **ViRanker** (XLM-RoBERTa based) model as the foundation for the DeeperImpact retrieval stage.

- **Negative Mining**: `filter_and_mine_hard_negatives.ipynb` pre-processes the corpus and mines hard negatives via BM25 to ensure robust contrastive learning.

- **Scoring**: Notebooks like `score_viranker_maxp.ipynb` generate relevance scores for the retrieval candidates.

### 3. Evaluation

Focused on benchmarking the performance of individual components and the hybrid system.

- **Retriever Benchmarks**: `evaluate_bm25_test.ipynb` and `evaluate_deeperimpact.ipynb` calculate metrics such as **MRR@10** and **Recall@k**.

- **Hybrid Evaluation**: `evaluate_hybrid_score.ipynb` tests the weighted alpha combination of lexical (BM25) and semantic scores.

- **Visualization**: `plot_final_results.ipynb` generates charts to compare system performance across different retrieval methods.

### 🚀 Root Notebooks

- `demo_final.ipynb`: The primary entry point for users. It launches a **Gradio web interface** that integrates the full pipeline—Hybrid retrieval and reasoning via the Gemini 2.5 Flash reader module.

## 🛠️ Environment & Setup

Most notebooks are designed to run in **Google Colaboratory** and include cells to mount **Google Drive** and install necessary dependencies. Key libraries used include:

- **Retrieval**: `pyserini` (Lucene-based BM25), `rank_bm25`.

- **NLP**: `underthesea` and `py_vncorenlp` for Vietnamese word segmentation.

- **Models**: `transformers`, `peft`, `torch` for ViRanker and DeeperImpact.

- **UI**: `gradio` for the final system demonstration.

## 🔄 Recommended Workflow

To reproduce the system results, it is recommended to follow this sequence:

1. **EDA & Pre-processing**: Run notebooks in `/Data Handling` to understand and segment the corpus.

2. **Negative Mining**: Execute `/Main Pipeline/filter_and_mine_hard_negatives.ipynb` to prepare the training environment.

3. **Model Training**: Use the training scripts in `/Main Pipeline` to fine-tune the semantic retriever.

4. **Evaluation**: Benchmark the trained models using the notebooks in `/Evaluation`.

5. **Inference**: Run `demo_final.ipynb` to test the end-to-end RAG system with live claims.