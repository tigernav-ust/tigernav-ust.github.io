import torch
from transformers import pipeline
import pandas as pd

# Load Zero-Shot Classification Model
model_name = "facebook/bart-large-mnli"  # Alternative: "facebook/bart-large-mnli"
classifier = pipeline("zero-shot-classification", model=model_name, device=0)  # 0 means first GPU

# Load Parquet dataset
parquet_file_path = 'Dataset\\Evaluation of Unstructured and Unconventional\\qna_pairs1.parquet'
df = pd.read_parquet(parquet_file_path)

# Define labels for classification
candidate_labels = ["Structured", "Unstructured", "Unconventional", "Unconventional and Unstructured", "Unconventional and Structured"]

# Classification function
def classify_query(text):
    result = classifier(text, candidate_labels)
    predicted_class = result["labels"][0]  # The highest confidence label
    return predicted_class

# Apply classification and collect the results
predictions = []
for index, row in df.iterrows():
    query = row['question']  # Assuming 'question' contains the text
    classification = classify_query(query)
    predictions.append(classification)

# Add the predicted classifications as a new column in the DataFrame
df['Predicted_Label'] = predictions

# Calculate the tally for each classification
tally = df['Predicted_Label'].value_counts()

# Create a DataFrame to store the classification results (Query, Classification)
result_df = df[['question', 'Predicted_Label']]
result_df.columns = ['Query', 'Classification']

# Add the tally as a summary at the bottom of the DataFrame
summary = pd.DataFrame(tally).reset_index()
summary.columns = ['Classification', 'Quantity']

# Concatenate the result_df with the summary DataFrame
result_df = pd.concat([result_df, summary], ignore_index=True)

# Save the results and tally to an Excel file
with pd.ExcelWriter('Classification_Output_with_Tally.xlsx') as writer:
    result_df.to_excel(writer, sheet_name='Query_Classification', index=False)

print("Classification results and tally saved to 'Classification_Output_with_Tally.xlsx'.")
