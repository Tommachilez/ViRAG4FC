import csv
import json
import time
import argparse
import os
import google.generativeai as genai
from tqdm import tqdm

def build_prompt(context: str, evidence: str, claim: str) -> str:
    """Constructs the exact prompt required for the task."""
    return f"""
User Claim (The statement being verified):
\"\"\"
{claim}
\"\"\"

Input Context (Passage):
\"\"\"
{context}
\"\"\"

Core Evidence (The specific sentences supporting/refuting the claim):
\"\"\"
{evidence}
\"\"\"

Task: Generate exactly 3 distinct Vietnamese search queries. 
CRITICAL INSTRUCTION: The queries must bridge the gap between the "User Claim" and the "Input Context". 
- Users will likely search using entities found in the "Claim" (e.g., names, dates). 
- However, the query must be answerable by the "Input Context". 

Categories: 
1. Category "KEYWORD" (Tìm kiếm từ khóa): 
- Simulate a user typing into Google. Use short, telegraphic keywords. 
- EXTRACT entities from the "Claim" (Who/What/When/Where) and combine them with terms from the "Context".
- NO grammar, NO question words (e.g., "là gì", "như thế nào").
2. Category "NATURAL" (Câu hỏi tự nhiên): 
- Simulate a user asking a voice assistant or chatbot. 
- Must be a complete grammatical sentence ending with a question mark. 
- The question should ask about the veracity of the "Claim" based on the "Context".
3. Category "SEMANTIC" (Biến thể Hán-Việt/Đồng nghĩa): 
- Simulate a user using different vocabulary than the text. 
- CRITICAL: Swap words from the "Claim" OR "Context" with Sino-Vietnamese (Hán-Việt) equivalents or synonyms (e.g., change "đất ở" -> "thổ cư", "người đứng đầu" -> "thủ trưởng", "sửa đổi" -> "tu chính").

CRITICAL: The queries must target the information found in the "Core Evidence", but the queries will be used to retrieve the full "Input Context".

Constraints:
- All output must be in valid JSON format. 
- Queries must be strictly in Vietnamese. 
- Do not hallucinate information not present in the text. 
- If the text is too short or meaningless to generate queries, return an empty JSON list "[]".

Output JSON Format: [ {{ "query": "...", "type": "KEYWORD" }}, {{ "query": "...", "type": "NATURAL" }}, {{ "query": "...", "type": "SEMANTIC", "reasoning": "Briefly explain which word you swapped (e.g., swapped 'vợ chồng' with 'phu thê')" }} ]
"""

def count_csv_rows(filepath):
    """Helper to count rows for the progress bar."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f) - 1 # Subtract header
    except:
        return None

def process_csv(args):
    # Configure API
    genai.configure(api_key=args.api_key)

    model = genai.GenerativeModel(
        model_name=args.model,
        generation_config={
            "temperature": 0.2,
            "response_mime_type": "application/json"
        }
    )

    print(f"Reading from: {args.input}")
    print(f"Writing to: {args.output}")

    # Determine total for progress bar
    total_rows = args.quota if args.quota else count_csv_rows(args.input)

    processed_count = 0

    # Check for existing output file to append vs write
    mode = 'a' if os.path.exists(args.output) and args.append else 'w'

    try:
        with open(args.input, mode='r', encoding='utf-8') as infile, \
             open(args.output, mode=mode, encoding='utf-8') as outfile:

            reader = csv.DictReader(infile)

            # Verify headers - checking for 'document' as per your notebook (was 'context' in original prompt)
            # We accept either 'document' or 'context' for flexibility
            headers = reader.fieldnames
            context_col = 'document' if 'document' in headers else 'context'
            claim_col = 'query' if 'query' in headers else 'claim'

            if context_col not in headers or 'evidence' not in headers or 'id' not in headers or claim_col not in headers:
                print(f"Error: CSV must contain columns: 'id', 'evidence', '{claim_col}', and '{context_col}'")
                return

            # Initialize tqdm progress bar
            pbar = tqdm(total=total_rows, desc="Generating Queries", unit="row")

            for row in reader:
                # Quota check
                if args.quota and processed_count >= args.quota:
                    break

                row_id = row.get('id')
                context = row.get(context_col, '')
                evidence = row.get('evidence', '')
                claim = row.get(claim_col, '')

                # Skip invalid rows
                if not context or not evidence:
                    continue
                
                prompt = build_prompt(context, evidence, claim)

                try:
                    response = model.generate_content(prompt)
                    generated_json = json.loads(response.text)

                    result_entry = {
                        "id": row_id,
                        "generated_queries": generated_json
                    }

                    outfile.write(json.dumps(result_entry, ensure_ascii=False) + "\n")
                    outfile.flush()

                    processed_count += 1
                    pbar.update(1)

                    # Rate limiting sleep
                    time.sleep(1)

                except Exception as e:
                    pbar.write(f"Error processing ID {row_id}: {e}")

            pbar.close()

    except FileNotFoundError:
        print(f"Error: File {args.input} not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    print(f"\nDone! Processed {processed_count} items.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate queries from CSV using Gemini API")

    # Required arguments
    parser.add_argument("--api_key", required=True, help="Your Gemini API Key")
    parser.add_argument("--input", required=True, help="Path to input CSV file")

    # Optional arguments
    parser.add_argument("--output", default="output_queries.jsonl", help="Path to output JSONL file")
    parser.add_argument("--quota", type=int, default=None, help="Number of rows to process (default: all)")
    parser.add_argument("--model", default="gemini-2.5-flash", help="Model name (default: gemini-1.5-flash)")
    parser.add_argument("--append", action="store_true", help="Append to output file instead of overwriting")

    args = parser.parse_args()

    process_csv(args)
