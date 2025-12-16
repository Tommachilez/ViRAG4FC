import csv
import json
import argparse
from tqdm import tqdm

# --- 1. Copy your existing prompt builder here ---
def build_prompt(context: str, evidence: str, claim: str) -> str:
    return f"""
Claim: \"\"\" {claim} \"\"\"
Context: \"\"\" {context} \"\"\"
Evidence: \"\"\" {evidence} \"\"\"

Task: Generate 3 Vietnamese search queries bridging the "Claim" entities and "Context" vocabulary. Queries must target the "Evidence" but retrieve the full "Context". Users will likely search using entities found in the "Claim" (e.g., names, dates).

Categories: 
1. "KEYWORD" (Tìm kiếm từ khóa): Use short, telegraphic keywords. EXTRACT entities from the "Claim" (Who/What/When/Where) combined with "Context" terms. NO grammar, NO question words (e.g., "là gì", "như thế nào").
2. "NATURAL" (Câu hỏi tự nhiên): 
- Simulate a user asking a voice assistant or chatbot. Must be a complete grammatical sentence ending with a question mark.
- PREFERRED: Use "Wh-questions" (Ai, Cái gì, Ở đâu, Khi nào, Tại sao) to seek specific facts from the Context. The goal is to ask for the EVIDENCE that supports/refutes the claim based on the "Context".
- AVOID: Simple Yes/No questions (e.g., avoid "Có phải...", "Đúng không") unless they significantly paraphrase the claim.
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

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate prompt batch JSONL from CSV")

    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument("--output", default="output_queries.jsonl", help="Path to output JSONL file")

    args = parser.parse_args()

    # --- 2. Build the JSONL file ---
    print(f"Converting {args.input} to Batch JSONL format...")

    rows_processed = 0
    with open(args.input, 'r', encoding='utf-8') as f_in, \
        open(args.output, 'w', encoding='utf-8') as f_out:

        reader = csv.DictReader(f_in)

        for row in tqdm(reader, desc="Preparing Requests"):
            row_id = str(row.get('id'))
            context = row.get('document') or row.get('context', '')
            evidence = row.get('evidence', '')
            claim = row.get('query') or row.get('claim', '')

            if not context or not evidence:
                continue

            # Create the prompt
            prompt_text = build_prompt(context, evidence, claim)

            # Construct the Batch Request Object
            # This wrapper tells the Batch API what to do for this specific row
            batch_request = {
                "custom_id": row_id,  # CRITICAL: This lets us match the answer back to the CSV later
                "method": "generateContent",
                "body": {
                    "contents": [
                        {"role": "user", "parts": [{"text": prompt_text}]}
                    ],
                    "generationConfig": {
                        "temperature": 0.2,
                        "response_mime_type": "application/json"
                    }
                }
            }

            # Write as a single line of JSON
            f_out.write(json.dumps(batch_request) + "\n")
            rows_processed += 1

    print(f"Done! Created {args.output} with {rows_processed} requests.")
