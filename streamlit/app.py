
import streamlit as st # type: ignore
import whisper # type: ignore
import torch # type: ignore
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler # type: ignore
import sounddevice as sd # type: ignore
from scipy.io.wavfile import write # type: ignore
import numpy as np # type: ignore

# Load Whisper and Stable Diffusion models
@st.cache_resource  
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

# Record and transcribe 
if st.button("Record"):
    st.write("Recording...")
    fs = 44100  
    audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()  
    st.write("Recording complete!")

    audio_path = "recorded_audio.wav"
    write(audio_path, fs, audio_data)

    st.write("Transcribing audio with Whisper model...")
    transcription = whisper_model.transcribe(audio_path)
    st.write(f"Transcription: {transcription['text']}")

    # Generate image
    st.write("Generating image with Stable Diffusion...")
    generated_image = pipe(transcription["text"]).images[0]

    st.image(generated_image, caption="Generated Image")
