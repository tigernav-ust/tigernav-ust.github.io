import pandas as pd
import json
import os

# Load the Excel file
# excel_file = "Dataset\SFT\Dataset.xlsx"
excel_file = "Dataset\Ground Truth\Ground Truth.xlsx"

# Dynamically get sheet names excluding "Checklist" and "Ruano Blueprint"
sheet_names = [sheet for sheet in pd.ExcelFile(excel_file).sheet_names if sheet not in ["Checklist", "Ruano Blueprint"]]

# Create a folder named 'compiled' if it doesn't already exist
output_folder = "Cosine Similarity\Ground Truth Dataset"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Loop through the sheets and process each one
for sheet_name in sheet_names:
    # Read the Excel file into a DataFrame for the current sheet, skipping the first two rows
    df = pd.read_excel(excel_file, sheet_name=sheet_name, header=None, skiprows=2)
    
    # Create a list to store the questions and answers for the current sheet
    qna_list = []
    
    # Loop through the columns, assuming the pattern is Question | Answer | [blank] | Question | Answer | [blank]...
    for i in range(0, len(df.columns), 3):  # Step through every third column
        if i + 1 < len(df.columns):  # Check if the next column exists
            question_col = df.iloc[:, i]       # The first column in the group (Question)
            answer_col = df.iloc[:, i + 1]     # The second column in the group (Answer)
            
            # Iterate over the rows and capture question-answer pairs
            for question, answer in zip(question_col, answer_col):
                if pd.notna(question) and pd.notna(answer):  # Only process non-empty pairs
                    qna_list.append({'Question': question, 'Answer': answer})
    
    # Save the data to a separate JSON file for each sheet
    json_file = os.path.join(output_folder, f"{sheet_name}.json")
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(qna_list, f, ensure_ascii=False, indent=4)

print(f"Data has been successfully converted and saved in the '{output_folder}' folder.")
