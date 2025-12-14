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
Claim:
\"\"\"
{claim}
\"\"\"

Context:
\"\"\"
{context}
\"\"\"

Evidence:
\"\"\"
{evidence}
\"\"\"

Task: Generate 3 Vietnamese search queries bridging the "Claim" entities and "Context" vocabulary. Queries must target the "Evidence" but retrieve the full "Context". Users will likely search using entities found in the "Claim" (e.g., names, dates).

Categories: 
1. "KEYWORD" (Tìm kiếm từ khóa): Use short, telegraphic keywords. EXTRACT entities from the "Claim" (Who/What/When/Where) combined with "Context" terms. NO grammar, NO question words (e.g., "là gì", "như thế nào").
2. "NATURAL" (Câu hỏi tự nhiên): 
- Simulate a user asking a voice assistant or chatbot.
- Must be a complete grammatical sentence ending with a question mark.
- PREFERRED: Use "Wh-questions" (Ai, Cái gì, Ở đâu, Khi nào, Tại sao) to seek specific facts from the Context.
- AVOID: Simple Yes/No questions (e.g., avoid "Có phải...", "Đúng không") unless they significantly paraphrase the claim.
- The goal is to ask for the EVIDENCE that supports/refutes the claim based on the "Context".
3. Category "SEMANTIC" (Biến thể Hán-Việt/Đồng nghĩa): 
- Simulate a user using different vocabulary than the text. 
- CRITICAL: Swap words from the "Claim" OR "Context" with Sino-Vietnamese (Hán-Việt) equivalents or synonyms (e.g., change "đất ở" -> "thổ cư", "sửa đổi" -> "tu chính").

Constraints:
- All output must be in valid JSON format. 
- Queries must be strictly in Vietnamese. 
- Do not hallucinate information not present in the text. 
- If the text is too short or meaningless to generate queries, return an empty JSON list "[]".

Output JSON: [ {{ "query": "...", "type": "KEYWORD" }}, {{ "query": "...", "type": "NATURAL" }}, {{ "query": "...", "type": "SEMANTIC", "reasoning": "Briefly explain which word you swapped (e.g., swapped 'vợ chồng' with 'phu thê')" }} ]
"""

def count_csv_rows(filepath):
    """Helper to count rows in input file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f) - 1 # Subtract header
    except:
        return 0

def get_existing_ids(output_path):
    """Reads the output file to find IDs that have already been processed."""
    existing_ids = set()
    if os.path.exists(output_path):
        print(f"Scanning {output_path} for existing records...")
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    line = line.strip()
                    if not line: continue
                    data = json.loads(line)
                    if 'id' in data:
                        existing_ids.add(str(data['id'])) # Ensure ID is string for comparison
                except json.JSONDecodeError:
                    continue
    return existing_ids

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

    # 1. Logic to handle Resuming
    existing_ids = set()
    
    # If appending, load existing work to skip
    if args.append and os.path.exists(args.output):
        existing_ids = get_existing_ids(args.output)
        print(f"Found {len(existing_ids)} items already processed. They will be skipped.")
        write_mode = 'a'
    else:
        # If not appending, we overwrite, so we start fresh
        print("Starting fresh (Overwrite mode).")
        write_mode = 'w'

    # 2. Calculate Totals for Progress Bar
    total_csv_rows = count_csv_rows(args.input)

    # Rows remaining to be done in the file
    remaining_rows = total_csv_rows - len(existing_ids)

    # Calculate how many we will actually process this run (limited by quota)
    if args.quota:
        rows_to_process_this_run = min(args.quota, remaining_rows)
    else:
        rows_to_process_this_run = remaining_rows

    if rows_to_process_this_run <= 0:
        print("No new rows to process! (Quota reached or file complete)")
        return

    processed_count = 0

    try:
        with open(args.input, mode='r', encoding='utf-8') as infile, \
             open(args.output, mode=write_mode, encoding='utf-8') as outfile:

            reader = csv.DictReader(infile)

            # Header mapping
            headers = reader.fieldnames
            context_col = 'document' if 'document' in headers else 'context'
            claim_col = 'query' if 'query' in headers else 'claim'

            # Header Validation
            if context_col not in headers or 'evidence' not in headers or 'id' not in headers or claim_col not in headers:
                print(f"Error: CSV must contain columns: 'id', 'evidence', '{claim_col}', and '{context_col}'")
                return

            # Initialize tqdm
            pbar = tqdm(total=rows_to_process_this_run, desc="Generating Queries", unit="row")

            for row in reader:
                # Quota Check (Stop if we have processed enough NEW items)
                if args.quota and processed_count >= args.quota:
                    break
                
                row_id = str(row.get('id'))

                # --- SKIP LOGIC ---
                # If this ID is already in the output file, skip it
                if row_id in existing_ids:
                    continue

                context = row.get(context_col, '')
                evidence = row.get('evidence', '')
                claim = row.get(claim_col, '')

                if not context or not evidence:
                    continue
                
                # Build Prompt
                prompt = build_prompt(context, evidence, claim)

                try:
                    # API Call
                    response = model.generate_content(prompt)
                    generated_json = json.loads(response.text)

                    result_entry = {
                        "id": row_id,
                        "generated_queries": generated_json
                    }

                    # Write to File
                    outfile.write(json.dumps(result_entry, ensure_ascii=False) + "\n")
                    outfile.flush()

                    # Update Counters
                    processed_count += 1
                    pbar.update(1)

                    # Rate Limit Sleep (Dynamic based on context length roughly)
                    sleep_time = 12 if len(context) > 750 else 2
                    time.sleep(sleep_time)

                except Exception as e:
                    error_str = str(e)
                    # Check for Quota Exceeded Error
                    if "429" in error_str or "Resource has been exhausted" in error_str:
                        pbar.write(f"\n[STOPPING] Quota exceeded at ID {row_id}. Saving progress and exiting.")
                        pbar.write(f"Error message: {error_str}")
                        break  # This stops the loop gracefully
                    else:
                        pbar.write(f"Error processing ID {row_id}: {e}")
                        # We do NOT increment processed_count here so we retry it next time script runs

            pbar.close()

    except FileNotFoundError:
        print(f"Error: File {args.input} not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    print(f"\nRun Complete! Processed {processed_count} new items.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate queries from CSV using Gemini API")

    parser.add_argument("--api_key", required=True, help="Your Gemini API Key")
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument("--output", default="output_queries.jsonl", help="Path to output JSONL file")
    parser.add_argument("--quota", type=int, default=None, help="Number of NEW rows to process in this run")
    parser.add_argument("--model", default="gemini-2.5-flash", help="Model name (default: gemini-2.5-flash)")
    parser.add_argument("--append", action="store_true", help="Resume/Append to output file. Required for skipping existing IDs.")

    args = parser.parse_args()

    process_csv(args)
