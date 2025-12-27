import time
import json
import google.generativeai as genai
from typing import List, Dict, Any

class ReaderLLM:
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        """
        Initialize the Gemini Reader with JSON output capabilities.
        :param api_key: The Google Gemini API key.
        :param model_name: The model version to use (default: gemini-2.5-flash).
        """
        if not api_key:
            raise ValueError("API Key must be provided to ReaderLLM.")

        genai.configure(api_key=api_key)
        self.model_name = model_name
        
        # Configure model for JSON output
        self.model = genai.GenerativeModel(
            model_name=model_name,
            generation_config={
                "temperature": 0.1, # Low temperature for factual consistency
                "response_mime_type": "application/json"
            }
        )

        # System prompt enforcing the "Judge" persona and JSON schema
        self.system_instruction = """
            You are a strict Verification Assistant for a Fact-Checking system.
            Your goal is to verify the user's claim using ONLY the provided context snippets.

            Output strictly in JSON format with the following keys:
            1. "qid": The exact query ID provided in the prompt.
            2. "query": The original claim/query text.
            3. "verdict": One of the following exact strings: "Supported", "Refuted", "Not Enough Information".
            4. "explanation": A detailed reasoning.
               - Evidence-Based Only: Do not use outside knowledge. If the answer is not in the context, set verdict to "Not Enough Information".
               - Citation Required: Every sentence in the explanation must be supported by a citation in the format [Doc ID]. 
                 (Example: "VinFast broke ground on July 28 [289].")
               - Handle Contradictions: If documents contradict, explain both sides.
        """

    def format_context(self, retrieved_docs: List[Dict]) -> str:
        """
        Formats retrieved documents into a labeled string for the LLM.
        Input: List of dicts {'id': '123', 'content': 'text...', 'score': 0.95}
        """
        context_str = ""
        for i, doc in enumerate(retrieved_docs):
            # Use explicit ID if available, otherwise index
            doc_id = doc.get('id', str(i+1))
            content = doc.get('content', doc.get('document', '')).strip()
            context_str += f"--- Document [{doc_id}] ---\n{content}\n\n"
        return context_str

    def generate_answer(self, qid: str, query: str, retrieved_docs: List[Dict]) -> Dict[str, Any]:
        """
        Generates the structured JSON answer with citations, including retry logic.
        
        :param qid: The ID of the query (required for output JSON).
        :param query: The user's claim or question.
        :param retrieved_docs: List of document dictionaries.
        :return: Dictionary containing {qid, query, verdict, explanation}.
        """

        # Default response for empty context
        if not retrieved_docs:
            return {
                "qid": qid,
                "query": query,
                "verdict": "Not Enough Information",
                "explanation": "No relevant documents were found to verify this claim."
            }

        # 1. Prepare Context
        context_block = self.format_context(retrieved_docs)

        # 2. Construct Prompt
        full_prompt = f"""{self.system_instruction}

            INPUT DATA:
            Query ID: "{qid}"
            User Claim: "{query}"

            RETRIEVED CONTEXT:
            {context_block}
        """

        # 3. Call Gemini with Retry Logic (aligned with reference script)
        max_retries = 5
        retry_delay = 10  # Start with 10 seconds

        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(full_prompt)

                # Parse JSON response
                result_json = json.loads(response.text)

                # Sanity check: Ensure keys exist, strictly necessary for downstream tasks
                if "verdict" not in result_json:
                    result_json["verdict"] = "Not Enough Information"
                    result_json["explanation"] = "Error: Model failed to generate a verdict."

                return result_json

            except Exception as e:
                error_str = str(e)
                # Check for rate limit errors (429 or Resource Exhausted)
                if "429" in error_str or "Resource has been exhausted" in error_str:
                    if attempt == max_retries - 1:
                        print(f"Error: Failed ID {qid} after {max_retries} retries due to rate limiting.")
                        return {
                            "qid": qid,
                            "query": query,
                            "verdict": "Not Enough Information",
                            "explanation": f"API Rate Limit Reached. Error: {e}"
                        }

                    # Exponential backoff
                    wait_time = retry_delay * (2 ** attempt)  # 10s, 20s, 40s...
                    print(f"Warning: Hit rate limit for ID {qid}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    # Non-retryable error (e.g., JSON parse error or safety filter)
                    print(f"Error generating answer for ID {qid}: {e}")
                    return {
                        "qid": qid,
                        "query": query,
                        "verdict": "Not Enough Information",
                        "explanation": f"Generation error: {e}"
                    }

        return {
            "qid": qid,
            "query": query,
            "verdict": "Not Enough Information",
            "explanation": "Unknown failure in generation loop."
        }
