import streamlit as st
import whisper
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np

# Load Whisper and Stable Diffusion models
@st.cache_resource  # Cache models to prevent reloading each time
def load_models():
    whisper_model = whisper.load_model("base")  # Load Whisper model for transcription
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")  # Load Stable Diffusion model
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    return whisper_model, pipe

whisper_model, pipe = load_models()

# Streamlit UI
st.title("Audio-to-Image Generator with Whisper & Stable Diffusion")

st.write("Record an audio input and generate an image based on the transcribed text.")

# Define recording duration
duration = st.slider("Select duration of recording (seconds):", 1, 10, 3)

# Record and transcribe when the "Record" button is clicked
if st.button("Record"):
    st.write("Recording...")
    fs = 44100  # Sample rate for audio recording
    audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()  # Wait until recording finishes
    st.write("Recording complete!")

    # Save the recorded audio to a .wav file
    audio_path = "recorded_audio.wav"
    write(audio_path, fs, audio_data)

    # Step 1: Transcribe audio with Whisper
    st.write("Transcribing audio with Whisper model...")
    transcription = whisper_model.transcribe(audio_path)
    st.write(f"Transcription: {transcription['text']}")

    # Step 2: Generate image based on transcription
    st.write("Generating image with Stable Diffusion...")
    generated_image = pipe(transcription["text"]).images[0]

    # Display the generated image
    st.image(generated_image, caption="Generated Image")
    


