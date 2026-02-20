import pandas as pd
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import evaluate
import os

# Load the fine-tuned model and tokenizerD:\code\Trained Model\Trainer\fine_tuned_gpt2_epoch_5_newdata
model_name = "Trained Model\\Trainer\\fine_tuned_gpt2medium_epoch_1"  # Replace with your model name
# model_name = "Trained Model\\SFT + DPO\\fine_tuned_gpt2_epoch_5"  # Replace with your model name
# model_name = "Trained Model\\ORPO\\finetuned_ORPO_Epoch_4"  # Replace with your model name

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Load the ground truth dataset
ground_truth_file = "Dataset\\Evaluation of Unstructured and Unconventional\\structured.parquet"  # Path to your Parquet file
ground_truth_df = pd.read_parquet(ground_truth_file)

# Extract questions and reference answers
questions = ground_truth_df['question'].tolist()
references = ground_truth_df['answer'].tolist()

# Generate predictions
predictions = []
for question in questions:
    inputs = tokenizer(question, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs['input_ids'].to(model.device)
    attention_mask = inputs['attention_mask'].to(model.device)
  
    with torch.no_grad():
        output = model.generate(input_ids, attention_mask=attention_mask, max_length=128)  # Adjust max_length as needed
  
    # Decode the output to get the prediction
    prediction = tokenizer.decode(output[0], skip_special_tokens=True)
  
    # Assuming the model generates the answer after the question, you might need to split the text
    # This is a simple heuristic; adjust based on your model's behavior
    answer_start = len(question)
    prediction_answer = prediction[answer_start:].strip()
  
    predictions.append(prediction_answer)

# Save predictions to a JSON file
predictions_data = [
    {"question": q, "reference": r, "prediction": p}
    for q, r, p in zip(questions, references, predictions)
]

# predictions_json_path = "Performance Metric (LLM)\\Trainer\\predictions_5.json"
# predictions_json_path = "Performance Metric (LLM)\\SFT + DPO\\predictions_5.json"
predictions_json_path = "Performance Metric (LLM)\\Trainer\\Expanded Dataset\\final_predictions_1_structured.json"

os.makedirs(os.path.dirname(predictions_json_path), exist_ok=True)
with open(predictions_json_path, 'w', encoding='utf-8') as f:
    json.dump(predictions_data, f, ensure_ascii=False, indent=4)

# Load evaluation metrics
bleu = evaluate.load("bleu")
meteor = evaluate.load("meteor")
rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")

# Evaluate predictions
bleu_results = bleu.compute(predictions=predictions, references=references)
meteor_results = meteor.compute(predictions=predictions, references=references)
rouge_results = rouge.compute(predictions=predictions, references=references)
bertscore_results = bertscore.compute(predictions=predictions, references=references, model_type='bert-base-uncased')

# Evaluate Perplexity
def calculate_perplexity(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs['input_ids'])
        loss = outputs.loss
    return torch.exp(loss).item()

perplexities = [calculate_perplexity(model, tokenizer, pred) for pred in predictions]
average_perplexity = sum(perplexities) / len(perplexities)

# Save evaluation results to a text file
# evaluation_results_path = "Performance Metric (LLM)\\Trainer\\evaluation_results_5.txt"
# evaluation_results_path = "Performance Metric (LLM)\\SFT + DPO\\evaluation_results_5.txt"
evaluation_results_path = "Performance Metric (LLM)\\Trainer\\Expanded Dataset\\final_evaluation_1_structured_structured.txt"

with open(evaluation_results_path, 'w', encoding='utf-8') as f:
    f.write(f"BLEU: {bleu_results['bleu']}\n")
    f.write(f"METEOR: {meteor_results['meteor']}\n")
    f.write(f"ROUGE: {rouge_results}\n")
    average_bertscore = sum(bertscore_results['f1']) / len(bertscore_results['f1'])
    f.write(f"BERTScore: {average_bertscore}\n")
    f.write(f"Average Perplexity: {average_perplexity}\n")

# Save evaluation results to an Excel file
# evaluation_results_excel_path = "Performance Metric (LLM)\\Trainer\\evaluation_results_5.xlsx"
# evaluation_results_excel_path = "Performance Metric (LLM)\\SFT + DPO\\evaluation_results_5.xlsx"
evaluation_results_excel_path = "Performance Metric (LLM)\\Trainer\\Expanded Dataset\\final_evaluation_results_1_structured.xlsx"

# Ensure the directory exists
os.makedirs(os.path.dirname(evaluation_results_excel_path), exist_ok=True)

results_df = pd.DataFrame({
    'Metric': ['BLEU', 'METEOR', 'BERTScore', 'Average Perplexity'],
    'Score': [
        bleu_results['bleu'],
        meteor_results['meteor'],
        average_bertscore,
        average_perplexity
    ]
})

# Save to Excel
results_df.to_excel(evaluation_results_excel_path, index=False)

print(f"Evaluation results saved to {evaluation_results_excel_path}")

