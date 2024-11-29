
import sounddevice as sd
import numpy as np
import whisper
import warnings
from diffusers import DiffusionPipeline

warnings.filterwarnings("ignore")

# Parameters for audio recording
duration = 5  # Duration in seconds
sample_rate = 16000  # Sample rate for audio recording

# Step 1: Record Audio
print("Recording...")

recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
sd.wait() 

print("Recording finished.")
recording = recording.flatten() 

# Step 2: Transcribe Audio with Whisper
print("Transcribing using Whisper...")

model = whisper.load_model("base")  
result = model.transcribe(recording, verbose=False)

# Extract the recognized text
prompt = result['text']
print("Recognized text:", prompt)

# Step 3: Use the Recognized Text as Prompt for Image Generation
print("Generating image from text prompt...")

model_id = "CompVis/ldm-text2im-large-256"
ldm = DiffusionPipeline.from_pretrained(model_id)

# Generate images from the transcribed text
images = ldm([prompt], num_inference_steps=50, eta=0.3, guidance_scale=6).images

# Save the generated images
for idx, image in enumerate(images):
    image.save(f"generated_image-{idx}.png")

print("Image generation complete.")
