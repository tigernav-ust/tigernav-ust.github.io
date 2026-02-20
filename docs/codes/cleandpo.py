import pandas as pd
import json
import re
import time
import pyarrow as pa
import pyarrow.parquet as pq
import csv  # Import the CSV module

def clean_text(text):
    """Clean text by removing unnecessary characters and whitespace."""
    if isinstance(text, str):
        # Remove quotation marks and strip whitespace
        text = re.sub(r'["]', '', text).strip()
    return text

# Load the JSON data
json_file_path = 'Dataset\DPO\DPO.json'  # Replace with your JSON file path
with open(json_file_path, 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)

# Measure the time taken for the whole process
start_time = time.time()

# Initialize a list to store cleaned entries
cleaned_data = []

# Process each entry in the JSON data
for entry in data:
    cleaned_entry = {
        "prompt": clean_text(entry['prompt']),
        "chosen": clean_text(entry['chosen']),
        "rejected": clean_text(entry['rejected']),
    }
    # Only add non-empty entries
    if cleaned_entry['prompt'] and cleaned_entry['chosen'] and cleaned_entry['rejected']:
        cleaned_data.append(cleaned_entry)

# Convert the cleaned data to a DataFrame
df = pd.DataFrame(cleaned_data)

# Save the DataFrame as a CSV file without extra quotation marks
csv_output_path = "Dataset\DPO\cleaned_dpo_data.csv"  # Define your output CSV file path
df.to_csv(
    csv_output_path,
    index=False,
    quoting=csv.QUOTE_NONE,  # Prevent quoting of fields
    escapechar='\\',  # Escape any special characters like commas
    encoding='utf-8'  # Ensure output encoding is correct
)

# Convert the DataFrame to an Arrow Table
table = pa.Table.from_pandas(df)

# Write the Arrow Table to a Parquet file
parquet_output_path = "Dataset\DPO\cleaned_dpo_data.parquet"  # Define your output Parquet file path
pq.write_table(table, parquet_output_path)

# Calculate the elapsed time
elapsed_time = time.time() - start_time
print("Elapsed time:", elapsed_time, "seconds")
