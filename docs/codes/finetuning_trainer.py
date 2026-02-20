# Importing necessary libraries
import json
import re
from pprint import pprint
import pandas as pd
import torch
from datasets import Dataset, load_dataset, DatasetDict
from huggingface_hub import notebook_login
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
import time
import os

# Suppress advisory warnings
os.environ["WANDB_DISABLE_INTERNAL_MESSAGES"] = "true"
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
os.environ["HF_TOKEN"] = "hf_<REDACTED>"

# Load the training dataset
train_parquet_file = "Dataset\\SFT\\train_df_test.parquet"
test_parquet_file = "Dataset\\SFT\\test_df_test.parquet"

# Load datasets
train_dataset = load_dataset("parquet", data_files=train_parquet_file)
test_dataset = load_dataset("parquet", data_files=test_parquet_file)

# Create a DatasetDict
dataset = DatasetDict({
    'train': train_dataset['train'],
    'test': test_dataset['train']
})

# Model that you want to train from the Hugging Face hub
model_name = "openai-community/gpt2-medium"
new_model = "Trained Model\\Trainer\\fine_tuned_gpt2_epoch_1_test"

# Record the start time
start_time = time.time()


# Set training parameters
training_arguments = TrainingArguments(
    output_dir="./Output Directory/Trainer/results_epoch1_test",
    num_train_epochs=1,
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
    # Create labels by shifting input_ids
    inputs = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)
    inputs["labels"] = inputs["input_ids"].copy()  # Set labels equal to input_ids
    return inputs

# Tokenize datasets
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_arguments,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
)

# Start training
trainer.train()

# Tokenize the evaluation prompt
eval_prompt = "Provide detailed directions from the Guard Post to Room 107, including any necessary turns or landmarks."
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
