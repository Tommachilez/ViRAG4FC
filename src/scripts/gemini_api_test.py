#!.venv/Scripts/python
import os
import argparse
import json
import time
from google import genai
from google.genai import types
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

try:
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
except AttributeError:
    print("Error: GEMINI_API_KEY not found. Please set it in your .env file.")
    exit()


def load_context_from_json(filepath):
    """Loads context data from a JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: The file '{filepath}' is not a valid JSON file.")
        return None


def extract_keywords_in_batches(data, batch_size=5):
    """
    Extracts keywords from context in batches and tracks progress.
    """
    if not data:
        print("No data to process.")
        return

    # Define the base prompt for the AI model
    base_prompt = (
        "Extract only the main keywords in the paragraph that are the most vital "
        "to its context and return with each word separated by a space in plain format, "
        "maximum 32 words. Return every keyword indicating the timing if present.\n\n"
        "Paragraph: "
    )

    results = {}
    
    # Process each item with a progress bar
    for key_id, context in tqdm(data, desc="Processing Contexts"):
        full_prompt = base_prompt + context

        try:
            # Make a single API call for the context
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_budget=0) # Disables thinking
                ),
            )

            # Process the response
            if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                keywords = response.candidates[0].content.parts[0].text.strip()
                results[key_id] = keywords
            else:
                results[key_id] = "No keywords found."

        except Exception as e:
            print(f"\nAn error occurred while processing {key_id}: {e}")
            results[key_id] = f"Error processing request: {e}"

    return results


def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Extract keywords from a JSON file using the Gemini API.")
    parser.add_argument(
        '-f', '--file-path',
        type=str,
        default='context.json',
        help='The path to the input JSON file containing context data. (default: context.json)'
    )
    parser.add_argument(
        '-n', '--num-context',
        type=int,
        default=None,
        help='The number of contexts to process from the file. Processes all by default.'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=5,
        help='The number of contexts to process in each API call batch. (default: 5)'
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    # Load context from the JSON file specified by the user
    context_data = load_context_from_json(args.file_path)

    if context_data:
        # Convert the dictionary to a list of tuples (key_id, context)
        data_items = list(context_data.items())

        # If --num-context is specified, slice the list to process only that many items
        if args.num_context is not None:
            data_items = data_items[:args.num_context]
            print(f"Processing the first {len(data_items)} contexts.")

        # Run the keyword extraction process with the specified batch size
        extracted_keywords = extract_keywords_in_batches(data_items, batch_size=args.batch_size)

        # Optionally, save the results to a file
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        # Extract the dataset name from the input file path to create a dynamic output filename
        base_filename = os.path.basename(args.file_path)
        # Assuming the format 'dataset_...'
        dataset_name = base_filename.split('_')[0] if '_' in base_filename else base_filename.rsplit('.', 1)[0]
        output_filename = f'data/{dataset_name}/extracted_keywords_{timestamp}.json'
        with open(output_filename, 'w', encoding='utf-8') as outfile:
            json.dump(extracted_keywords, outfile, indent=4)
        print(f"\nKeyword extraction complete. Results saved to '{output_filename}'")

if __name__ == "__main__":
    main()
