from transformers import AutoModelForCausalLM, LlamaTokenizer
from huggingface_hub import login
login(token="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")

# Specify the model you want to download
model_name = "linhvu/decapoda-research-llama-7b-hf"  # Model from Deca-Llama
# Download the model
print("Downloading model...")
model = AutoModelForCausalLM.from_pretrained(model_name)

# Download the tokenizer using the LlamaTokenizer class
print("Downloading tokenizer...")
tokenizer = LlamaTokenizer.from_pretrained(model_name)

# Save the model and tokenizer locally
print("Saving model and tokenizer locally...")
model.save_pretrained('./llama_model')
tokenizer.save_pretrained('./llama_tokenizer')

print("Model and tokenizer saved successfully.")
