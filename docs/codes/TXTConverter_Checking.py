import pandas as pd
import os

# List of Excel files to process
excel_files = ["Dataset/Evaluation of Unstructured and Unconventional/Unstructured_Unconventional.xlsx"]  # Adjust the path accordingly

# Output files
txt_file = "Dataset/Evaluation of Unstructured and Unconventional/class_with_labels.txt"
parquet_file = "Dataset/Evaluation of Unstructured and Unconventional/class_with_labels.parquet"

# Create an empty list to store the questions, answers, and labels
qna_list = []

# Loop through each Excel file
for excel_file in excel_files:
    # Get all sheet names except the ones to exclude
    sheet_names = [sheet for sheet in pd.ExcelFile(excel_file).sheet_names if sheet not in ["Checklist", "Ruano Blueprint"]]

    # Loop through the sheets and process each one
    for sheet_name in sheet_names:
        # Read the sheet, skipping the first two rows
        df = pd.read_excel(excel_file, sheet_name=sheet_name, header=None, skiprows=2)

        # Process columns in groups of three (Question | Answer | Label)
        for i in range(0, len(df.columns), 3):  
            if i + 2 < len(df.columns):  
                question_col = df.iloc[:, i]  
                answer_col = df.iloc[:, i + 1]
                label_col = df.iloc[:, i + 2]  # Third column is the label

                # Iterate over the rows and capture question-answer-label triplets
                for question, answer, label in zip(question_col, answer_col, label_col):
                    if pd.notna(question) and pd.notna(answer) and pd.notna(label):  
                        qna_list.append({'Question': question, 'Answer': answer, 'Label': label})

# Convert list to DataFrame
qna_df = pd.DataFrame(qna_list)

# Save to text file
with open(txt_file, 'w', encoding='utf-8') as f:
    for _, row in qna_df.iterrows():
        f.write(f"Question: {row['Question']}\nAnswer: {row['Answer']}\nLabel: {row['Label']}\n\n")

# Save to Parquet file
qna_df.to_parquet(parquet_file, index=False)

print(f"Data has been successfully converted to {txt_file} and {parquet_file}")
