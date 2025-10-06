#!.venv/Scripts/python
import csv
from datasets import load_dataset

dataset = load_dataset("tranthaihoa/vifactcheck")

all_rows = []
GLOBAL_ID = 0

# Iterate over each split in the dataset (e.g., 'train', 'validation', 'test')
for split_name, split_data in dataset.items():
    print(f"Processing split: '{split_name}'...")
    # 4. Enumerate through each item in the current split
    for item in split_data:
        # 5. Extract the required fields and create a dictionary for the row
        row = {
            'id': GLOBAL_ID,
            'statement': item['Statement'],
            'context': item['Context'],
            'label': item['labels']
        }
        all_rows.append(row)
        GLOBAL_ID += 1

# 6. Define the output CSV file name and the header
OUTPUT_FILE = 'data/vifactcheck_extract_data.csv'
header = ['id', 'statement', 'context', 'label']

# 7. Write the collected data into the CSV file
print(f"Writing data to '{OUTPUT_FILE}'...")
try:
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header)

        # Write the header row
        writer.writeheader()

        # Write all the data rows
        writer.writerows(all_rows)

    print(f"Successfully extracted {GLOBAL_ID} records and saved to '{OUTPUT_FILE}'.")

except IOError as e:
    print(f"Error writing to file: {e}")
