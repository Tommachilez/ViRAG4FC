import csv
import argparse
import os
from collections import defaultdict
from tqdm import tqdm
from reader_llm import ReaderLLM

def load_document_mapping(doc_mapping_path):
    """Loads the map of DocID -> Document Text."""
    print(f"Loading document text from {doc_mapping_path}...")
    doc_map = {}
    with open(doc_mapping_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc="Loading Docs"):
            if 'doc_id' in row and 'document' in row:
                doc_map[row['doc_id'].strip()] = row['document'].strip()
    return doc_map

def load_queries(query_mapping_path):
    """Loads the map of QueryID -> Query Text."""
    print(f"Loading queries from {query_mapping_path}...")
    queries = {}
    with open(query_mapping_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if 'query_id' in row and 'query' in row:
                queries[row['query_id'].strip()] = row['query'].strip()
    return queries

def load_run_file(run_file_path, top_k=5):
    """
    Parses the DeeperImpact run file to get top-K documents per query.
    Format: qid \t doc_id \t rank \t score
    """
    print(f"Loading top-{top_k} results from {run_file_path}...")
    retrieved_results = defaultdict(list)

    with open(run_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 4: continue

            qid = parts[0]
            doc_id = parts[1]
            rank = int(parts[2])
            score = float(parts[3])

            # Only keep top K docs
            if rank <= top_k:
                retrieved_results[qid].append((doc_id, score))

    return retrieved_results

def main():
    parser = argparse.ArgumentParser(description="Run RAG using DeeperImpact results and ReaderLLM.")

    # RAG Inputs
    parser.add_argument("--run_file", type=str, required=True, help="Path to run_file.txt from DeeperImpact")
    parser.add_argument("--doc_mapping", type=str, required=True, help="Path to unique_doc_mapping.csv")
    parser.add_argument("--query_mapping", type=str, required=True, help="Path to test_query_mapping.csv")
    parser.add_argument("--qid", type=str, help="Specific Query ID to process (optional). If not set, processes all.")
    parser.add_argument("--top_k", type=int, default=3, help="Number of documents to send to LLM")

    # LLM Configuration
    parser.add_argument("--api_key", type=str, help="Gemini API Key (overrides env var GEMINI_API_KEY)")
    parser.add_argument("--model_name", type=str, default="gemini-2.5-flash", help="Gemini model name (default: gemini-2.5-flash)")

    args = parser.parse_args()

    # Determine API Key
    api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: API Key must be provided via --api_key argument or GEMINI_API_KEY environment variable.")
        exit(1)

    # 1. Initialize Reader LLM
    print(f">>> Initializing Reader LLM with model: {args.model_name}...")
    try:
        reader = ReaderLLM(api_key=api_key, model_name=args.model_name)
    except Exception as e:
        print(f"Error initializing Gemini: {e}")
        exit(1)

    # 2. Load Data
    doc_map = load_document_mapping(args.doc_mapping)
    queries = load_queries(args.query_mapping)
    run_data = load_run_file(args.run_file, top_k=args.top_k)

    # 3. Determine targets
    target_qids = [args.qid] if args.qid else list(queries.keys())

    # 4. RAG Execution Loop
    for qid in target_qids:
        qid = str(qid)
        if qid not in queries:
            print(f"Warning: Query ID {qid} not found in query mapping.")
            continue

        query_text = queries[qid]
        retrieved_docs_ids = run_data.get(qid, [])

        # Format docs for ReaderLLM: List[Dict{'id', 'content'}]
        llm_input_docs = []
        for doc_id, score in retrieved_docs_ids:
            content = doc_map.get(doc_id)
            if content:
                llm_input_docs.append({
                    "id": doc_id,
                    "content": content,
                    "score": score
                })

        if not llm_input_docs:
            print(f"Skipping QID {qid}: No retrieved documents found.")
            continue

        print(f"\n{'='*60}")
        print(f"QUERY [{qid}]: {query_text}")
        print(f"RETRIEVED: {len(llm_input_docs)} documents")
        print(f"{'='*60}")

        # Generate Answer
        answer = reader.generate_answer(query_text, llm_input_docs)

        print(f"ANSWER:\n{answer}\n")

if __name__ == "__main__":
    main()
