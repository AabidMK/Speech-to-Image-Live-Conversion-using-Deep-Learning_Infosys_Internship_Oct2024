import streamlit as st
import whisper
import sounddevice as sd
import numpy as np
import torch
from diffusers import StableDiffusionPipeline

# Load the Whisper model
model = whisper.load_model("base")

# Function to record audio
def record_audio(duration=5, samplerate=16000):
    st.write("Recording...")
    audio = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()
    st.write("Recording finished.")
    return audio.flatten()

# Function to transcribe audio using Whisper
def transcribe_audio(audio):
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    options = whisper.DecodingOptions(language="en")
    result = model.decode(mel, options)
    return result.text

# Streamlit app
def main():
    st.title("Audio Transcription and Image Generation")
    
    # Record audio
    if st.button("Record Audio"):
        duration = st.slider("Select recording duration in seconds", 1, 10, 5)
        audio = record_audio(duration)
        text = transcribe_audio(audio)
        st.write(f"Recognized: {text}")
        # Load the stable diffusion model
        model_id = "CompVis/stable-diffusion-v1-4"
        pipe = StableDiffusionPipeline.from_pretrained(model_id)
        with torch.autocast("cuda"):
            image = pipe(text, guidance_scale=7.5).images[0]
        
        # Display the image
        st.image(image, caption="Generated Image", use_column_width=True)

if __name__ == "__main__":
    main()

