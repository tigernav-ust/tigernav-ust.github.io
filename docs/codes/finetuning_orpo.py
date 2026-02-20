import json
import os
import time
import torch
from datasets import Dataset, DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import ORPOTrainer, ORPOConfig

# Suppress advisory warnings
os.environ["WANDB_DISABLE_INTERNAL_MESSAGES"] = "true"
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
os.environ["HF_TOKEN"] = "hf_<REDACTED>"

# Load the JSON dataset
with open("Dataset/DPO/dpo_train.json", "r", encoding="utf-8") as f:
  train_data = json.load(f)

with open("Dataset/DPO/dpo_test.json", "r", encoding="utf-8") as f:
  test_data = json.load(f)

# Ensure the data is structured correctly
def process_data(data):
  processed_data = []
  for entry in data:
      prompt = entry.get("prompt", "").strip()
      chosen = entry.get("chosen", "").strip()
      rejected = entry.get("rejected", "").strip()
      if prompt and chosen and rejected:
          processed_data.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
  return processed_data

# Process the datasets
train_data_processed = process_data(train_data)
test_data_processed = process_data(test_data)

# Convert to DatasetDict
train_dataset = Dataset.from_list(train_data_processed)
test_dataset = Dataset.from_list(test_data_processed)

dataset = DatasetDict({
  'train': train_dataset,
  'test': test_dataset
})

# Model that you want to train from the Hugging Face hub
model_name = "openai-community/gpt2"
new_model = "Trained Model//ORPO//finetuned_ORPO_Epoch_4"

# Record the start time
start_time = time.time()

# Set training parameters using ORPOConfig
orpo_config = ORPOConfig(
  output_dir="./Output Directory/ORPO/ORPO_Epoch_4",
  num_train_epochs=4,
  per_device_train_batch_size=2,
  per_device_eval_batch_size=2,
  gradient_accumulation_steps=2,
  learning_rate=5e-5,
  weight_decay=0.01,
  max_grad_norm=1.0,
  lr_scheduler_type="linear",
  save_steps=2000,
  logging_steps=100,
  evaluation_strategy="epoch",
  beta=0.1  # Set the beta parameter for ORPO
)

# Load the GPT-2 model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Set padding token to eos token

# Load the model
model = AutoModelForCausalLM.from_pretrained(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Tokenize the training dataset
def tokenize_function(examples):
  # Tokenize each field separately
  prompt_inputs = tokenizer(examples["prompt"], truncation=True, padding="max_length", max_length=128)
  chosen_inputs = tokenizer(examples["chosen"], truncation=True, padding="max_length", max_length=128)
  rejected_inputs = tokenizer(examples["rejected"], truncation=True, padding="max_length", max_length=128)
  
  # Return the tokenized inputs along with the original fields
  return {
      "prompt": examples["prompt"],
      "chosen": examples["chosen"],
      "rejected": examples["rejected"],
      "prompt_input_ids": prompt_inputs["input_ids"],
      "prompt_attention_mask": prompt_inputs["attention_mask"],
      "chosen_input_ids": chosen_inputs["input_ids"],
      "chosen_attention_mask": chosen_inputs["attention_mask"],
      "rejected_input_ids": rejected_inputs["input_ids"],
      "rejected_attention_mask": rejected_inputs["attention_mask"],
  }

# Tokenize datasets
tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=dataset["train"].column_names)

# Initialize ORPOTrainer with tokenizer
orpo_trainer = ORPOTrainer(
  model=model,
  args=orpo_config,
  train_dataset=tokenized_datasets['train'],
  eval_dataset=tokenized_datasets['test'],
  tokenizer=tokenizer  # Add tokenizer here
)

# Start training
orpo_trainer.train()

# Tokenize the evaluation prompt
eval_prompt = "How can I navigate to Room 101 from Room 108?"
model_input = tokenizer(eval_prompt, return_tensors="pt").to(device)

# Set the model to evaluation mode
model.eval()

# Print training duration
end_time = time.time()
training_time = end_time - start_time
print(f"Training took {training_time} seconds")

# Save the fine-tuned model
model.save_pretrained(new_model)
tokenizer.save_pretrained(new_model)