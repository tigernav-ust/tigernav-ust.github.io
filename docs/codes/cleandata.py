import pandas as pd
import re
import time
import pyarrow as pa
import pyarrow.parquet as pq
import csv  # Import the CSV module

def extract_qa_pairs(text):
    qa_pairs = []
        # Update the regular expression to remove numbers, Q/A labels, and quotes
    text = re.sub(r'^[0-9]+[.]|q+[:]|a+[:]|Q+[0-9]*[:]|A+[0-9]*[:]|Question+[:]|Answer+[:]|["]', '', text, flags=re.MULTILINE).strip()
    
    # Split the text using two or more newlines as the separator for each Q&A pair
    entries = re.split(r'\n\s*\n', text.strip())
    
    for entry in entries:
        lines = entry.strip().split('\n')
        question, answer = "", ""
        
        for line in lines:
            if line.lower().startswith("question:"):
                question = line[len("question:"):].strip()
            elif line.lower().startswith("answer:"):
                answer = line[len("answer:"):].strip()
            else:
                # If a line follows a question or answer without a new label, assume it's a continuation
                if not answer:
                    question += " " + line.strip()
                else:
                    answer += " " + line.strip()
        
        if question and answer:
            qa_pairs.append({"question": question.strip(), "answer": answer.strip()})
    
    return qa_pairs

# Read the text file with correct encoding (try 'utf-8' or 'utf-8-sig')
with open("old\\new\\Dataset\\SFTtxt.txt", 'r', encoding='utf-8') as f:
    text = f.read()

# Split the text into sections using regular expressions
sections = re.split(r'\n\s*\n', text.strip())

# Measure the time taken for the whole process
start_time = time.time()

# Initialize an empty list to store Q&A pairs
qna_pairs = []
for section in sections:
    qna_pairs.extend(extract_qa_pairs(section))

# Filter out incomplete Q&A pairs
qna_pairs = [pair for pair in qna_pairs if pair["question"] and pair["answer"]]

# Convert the Q&A pairs to a DataFrame
df = pd.DataFrame(qna_pairs)

# Save the DataFrame as a CSV file without extra quotation marks
df.to_csv(
    "old\\new\\Dataset\\qna_pairs.csv",
    index=False,
    quoting=csv.QUOTE_NONE,  # Prevent quoting of fields
    escapechar='\\',  # Escape any special characters like commas
    encoding='utf-8'  # Ensure output encoding is correct
)

# Convert the DataFrame to an Arrow Table
table = pa.Table.from_pandas(df)

# Write the Arrow Table to a Parquet file
pq.write_table(table, "old\\new\\Dataset\\qna_pairs.parquet")

# Calculate the elapsed time
elapsed_time = time.time() - start_time
print("Elapsed time:", elapsed_time, "seconds")