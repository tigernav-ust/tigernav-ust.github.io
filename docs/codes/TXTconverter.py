import pandas as pd
import os

# List of Excel files to process
excel_files = ["Dataset\\SFT\\Dataset10.xlsx"
               , "Dataset\\SFT\\Dataset9.xlsx"]  # Add your Excel files here

# Output text file
txt_file = "Dataset\\SFT\\QnA_General.txt" 

# Create an empty list to store the questions and answers
qna_list = []

# Loop through each Excel file
for excel_file in excel_files:
    # Get all sheet names except the ones to exclude
    sheet_names = [sheet for sheet in pd.ExcelFile(excel_file).sheet_names if sheet not in ["Checklist", "Ruano Blueprint"]]

    # Loop through the sheets and process each one
    for sheet_name in sheet_names:
        # Read the sheet, skipping the first two rows
        df = pd.read_excel(excel_file, sheet_name=sheet_name, header=None, skiprows=2)

        # Process columns in groups of three (Question | Answer | [blank])
        for i in range(0, len(df.columns), 3):  
            if i + 1 < len(df.columns):  
                question_col = df.iloc[:, i]  
                answer_col = df.iloc[:, i + 1]  

                # Iterate over the rows and capture question-answer pairs
                for question, answer in zip(question_col, answer_col):
                    if pd.notna(question) and pd.notna(answer):  
                        qna_list.append({'Question': question, 'Answer': answer})

# Save the consolidated data to a text file
with open(txt_file, 'w', encoding='utf-8') as f:
    for qna in qna_list:
        f.write(f"Question: {qna['Question']}\nAnswer: {qna['Answer']}\n\n")

print(f"Data from all Excel files has been successfully converted to {txt_file}")
