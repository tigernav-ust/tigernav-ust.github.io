import pandas as pd
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
import pyarrow as pa
import time
import os

# Load Parquet file
parquet_file = 'Dataset\\SFT\\QnA_General.parquet'
df = pd.read_parquet(parquet_file)

def process_row(row):
    try:
        question = row['question']
        answer = row['answer']

        # Manually format the chat instead of using apply_chat_template
        tokenized_chat = f"{question}\n{answer}" 

        # tokenized_chat = f"User: {question}\nAssistant: {answer}"

        json_object = {
            "text": tokenized_chat,
            "instruction": "You are a friendly chatbot specializing in providing direction about the Fr. Roque Ruano Building.",
            "input": question,
            "output": answer
        }

        return json_object
    except Exception as e:
        print(f"Error processing row: {e}")
        return None

# Process DataFrame
start_time = time.time()
json_list = [process_row(row) for index, row in df.iterrows() if process_row(row) is not None]
end_time = time.time()

# Check if the processed list is empty
if not json_list:
    raise ValueError("No valid data was processed. Please check the input data.")

# Create new DataFrame from the processed JSON objects
new_df = pd.DataFrame(json_list)

# Split the dataset if it's not empty
if len(new_df) > 0:
    train_df, test_df = train_test_split(new_df, test_size=0.3, random_state=42)

    # Save to Parquet
    train_df.to_parquet('Dataset\\SFT\\train_df_test.parquet', index=False)
    test_df.to_parquet('Dataset\\SFT\\test_df_test.parquet', index=False)

    # Compute duration
    duration = end_time - start_time
    print("Duration:", duration, "seconds")
else:
    print("No data to split.")
