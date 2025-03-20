import os
from transformers import AutoModelForCausalLM, AutoTokenizer

def download_model(model_name: str, save_directory: str, huggingface_key: str):
    """Download a Hugging Face model with the tokenizer included."""
    try:
        print(f"Downloading {model_name}...")
        os.environ["HF_HOME"] = save_directory
        os.environ["HF_TOKEN"] = huggingface_key
        model_and_tokenizer = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=True)
        print(f"Successfully downloaded {model_name}")
    except Exception as e:
        print(f"Error downloading {model_name}: {e}")

if __name__ == "__main__":
    save_dir = os.path.expanduser("~/llama_models")
    os.makedirs(save_dir, exist_ok=True)

    huggingface_key = input("Enter your Hugging Face API key: ").strip()

    # List of models to download
    models = [
        "meta-llama/Llama-2-7b-hf",
        "meta-llama/Llama-2-13b-hf",
        "meta-llama/Llama-2-70b-hf",
    ]

    for model_name in models:
        download_model(model_name, save_dir, huggingface_key)

    print("All models downloaded.")
