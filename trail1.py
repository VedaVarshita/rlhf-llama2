
from trl import DPOTrainer, DPOConfig
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset

import pandas as pd
# Preprocess dataset: extract the 'prompt', 'chosen', and 'rejected' fields for training
def preprocess_hh(example):
    return {
        "chosen": example["chosen"],
        "rejected": example["rejected"]
    }

def tokenize(example):
    kwargs = {"padding": "max_length", "truncation": True, "max_length": 512, "return_tensors": "pt"}
    
    # Tokenize prompt + chosen and prompt + rejected responses
    chosen_tokens = tokenizer(example["query"] + "\n" + example["chosen"], **kwargs)
    rejected_tokens = tokenizer(example["query"] + "\n" + example["rejected"], **kwargs)
    
    return {
        "query": example["query"],
        "chosen_ids": chosen_tokens["input_ids"][0],
        "rejected_ids": rejected_tokens["input_ids"][0]
    }


# Define model and dataset
BASE_MODEL = "meta-llama/Llama-2-7b-hf"  # Example Llama-2 model
DATASET_NAME = "Anthropic/hh-rlhf"  # HH dataset from Hugging Face
OUTPUT_DIR = "./llama2_dpo_rlhf_model"

# Load the HHH dataset
dataset = load_dataset(DATASET_NAME)
print(dataset)


# Apply preprocessing and create train/test splits
processed_dataset = dataset.map(preprocess_hh)
# Split 10% of the training set for validation
train_valid_split = processed_dataset["train"].train_test_split(test_size=0.1, seed=42)

# Update train and validation datasets
train_dataset = train_valid_split["train"]
eval_dataset = train_valid_split["test"]

# Use the test set from the original dataset
test_dataset = processed_dataset["test"]


# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="auto")

# Add special tokens if needed
tokenizer.add_special_tokens({"pad_token": "[PAD]"})
model.resize_token_embeddings(len(tokenizer))




# Apply tokenization
train_dataset = train_dataset.map(tokenize, batched=False)
eval_dataset = eval_dataset.map(tokenize, batched=False)

# Remove unnecessary columns
train_dataset = train_dataset.remove_columns(["prompt", "chosen", "rejected"])
eval_dataset = eval_dataset.remove_columns(["prompt", "chosen", "rejected"])

# Configure training arguments
dpo_config = DPOConfig(
    model_name_or_path=BASE_MODEL,
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=5e-5,
    evaluation_strategy="steps",
    save_strategy="steps",
    save_steps=500,
    eval_steps=500,
    logging_steps=100,
    warmup_steps=100,
    gradient_accumulation_steps=8,
    save_total_limit=2,
    do_train=True,
    do_eval=True
)

# Define the trainer
trainer = DPOTrainer(
    model=model,
    tokenizer=tokenizer,
    args=dpo_config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)


# Train the model
trainer.train()

# Evaluate the model
trainer.evaluate()


# Save locally
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Optional: Push to Hugging Face Hub
from huggingface_hub import login
login()
trainer.push_to_hub("llama2_dpo_hhh_rlhf")
