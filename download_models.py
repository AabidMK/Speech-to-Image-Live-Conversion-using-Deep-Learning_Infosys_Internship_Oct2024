from transformers import WhisperForConditionalGeneration, WhisperProcessor

# Define the model to download
model_name = "openai/whisper-small"  # Replace with the correct model ID

# Define local cache directory
cache_dir = "./models"

# Download and cache the model and processor
print("Downloading and caching the Whisper model...")
whisper_model = WhisperForConditionalGeneration.from_pretrained(model_name, cache_dir=cache_dir)
whisper_processor = WhisperProcessor.from_pretrained(model_name, cache_dir=cache_dir)

print(f"Model downloaded and cached locally at {cache_dir}")
