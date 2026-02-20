import pandas as pd

# Replace 'your_file.parquet' with the path to your Parquet file
file_path = "Dataset\SFT\qna_pairs2.parquet"  # Use a raw string to avoid escape sequence issues

# Read the Parquet file
df = pd.read_parquet(file_path)

# Display the DataFrame
print(df)
