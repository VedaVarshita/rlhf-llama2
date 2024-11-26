from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
from trail1 import OUTPUT_DIR

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(OUTPUT_DIR)
tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)

# Create text-generation pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Example inference
prompt = "How can I make ethical decisions?"
response = pipe(prompt, max_length=200, num_return_sequences=1)
print(response)
