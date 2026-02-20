import pandas as pd
import json
import re

def clean_text(text):
    """Clean text by removing unnecessary characters, fixing apostrophes, and whitespace."""
    if isinstance(text, str):
        # Replace curly apostrophes with standard apostrophes
        text = re.sub(r'\u2019', "'", text)  # Fixes what’s to what's
        # Remove other unwanted characters and strip whitespace
        text = re.sub(r'["]', '', text).strip()
    return text

# Load the cleaned data from the Parquet file
cleaned_data_path = 'Dataset/DPO/cleaned_dpo_data.parquet'  # Path to your cleaned data
df = pd.read_parquet(cleaned_data_path)

# Convert DataFrame to a list of dictionaries and clean text fields
data = df.to_dict(orient='records')
for entry in data:
    entry['prompt'] = clean_text(entry['prompt'])
    entry['chosen'] = clean_text(entry['chosen'])
    entry['rejected'] = clean_text(entry['rejected'])

# Define the length for the test split (10% for testing)
test_len = int(len(data) * 0.1)

# Split the data into test and training sets
test = data[:test_len]
train = data[test_len:]

# Save the datasets as JSON files
with open('Dataset/DPO/dpo_train.json', 'w') as f:
    json.dump(train, f, indent=4)

with open('Dataset/DPO/dpo_test.json', 'w') as f:
    json.dump(test, f, indent=4)

# Save the test set as a Parquet file
df_test = pd.DataFrame(test)
df_test.to_parquet('Dataset/DPO/test.parquet')

# Print the number of entries in each dataset
print(f"Training set size: {len(train)}")
print(f"Test set size: {len(test)}")
