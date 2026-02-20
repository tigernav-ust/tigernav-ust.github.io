import pandas as pd
import json

# Load the Excel file
excel_file_path = 'Dataset/DPO/DPO.xlsx'  # Replace with your Excel file path

# Load all sheet names from the Excel file
sheet_names = pd.ExcelFile(excel_file_path).sheet_names

# Initialize a list to hold the Q&A pairs
qna_list = []

# Define sheets to exclude
sheets_to_exclude = {"Ruano Blueprint", "Checklist"}

# Loop through the sheets and process each one
for sheet_name in sheet_names:
    if sheet_name in sheets_to_exclude:
        print(f"Skipping sheet: {sheet_name}")
        continue  # Skip the excluded sheets

    # Read the Excel file into a DataFrame for the current sheet, skipping the first two rows
    df = pd.read_excel(excel_file_path, sheet_name=sheet_name, header=None, skiprows=2)

    # Loop through the columns, assuming the pattern is Question | Chosen | Rejected | Blank | Question | Chosen | Rejected...
    for i in range(0, len(df.columns), 4):  # Step through every fourth column
        # Check if the columns exist before accessing them
        if i + 2 < len(df.columns):  # Ensure there are at least three columns available
            question_col = df.iloc[:, i]          # The first column in the group (Question)
            chosen_col = df.iloc[:, i + 1]        # The second column in the group (Chosen)
            rejected_col = df.iloc[:, i + 2]      # The third column in the group (Rejected)

            # Iterate over the rows and capture question, chosen, rejected pairs
            for question, chosen, rejected in zip(question_col, chosen_col, rejected_col):
                if pd.notna(question) and pd.notna(chosen):  # Only process non-empty pairs
                    # Decode any special characters or encode them to prevent issues
                    entry = {
                        'prompt': str(question).encode('utf-8').decode('utf-8'),
                        'chosen': str(chosen).encode('utf-8').decode('utf-8'),
                        'rejected': str(rejected).encode('utf-8').decode('utf-8') if pd.notna(rejected) else None
                    }
                    qna_list.append(entry)
        else:
            print(f"Not enough columns in sheet '{sheet_name}' at index {i}")

# Convert the list of entries to JSON format
json_output_path = 'Dataset/DPO/DPO.json'  # Define your output JSON file path

with open(json_output_path, 'w', encoding='utf-8') as json_file:
    json.dump(qna_list, json_file, ensure_ascii=False, indent=4)

print(f"JSON file has been created at: {json_output_path}")
