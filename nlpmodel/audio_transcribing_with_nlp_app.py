import streamlit as st
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import librosa
import nltk
from nltk.stem import PorterStemmer

# Initialize stemmer
stemmer = PorterStemmer()

# Load Whisper and Stable Diffusion models
@st.cache_resource  # Cache models to prevent reloading each time
def load_models():
    # Load the fine-tuned Whisper model and processor from the specified directory
    processor = WhisperProcessor.from_pretrained("C:/Users/HP/OneDrive/Desktop/Infosys Springboard/whisper-finetuned")
    whisper_model = WhisperForConditionalGeneration.from_pretrained("C:/Users/HP/OneDrive/Desktop/Infosys Springboard/whisper-finetuned")
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")  # Load Stable Diffusion model
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    return whisper_model, processor, pipe

whisper_model, processor, pipe = load_models()

# Streamlit UI
st.title("Audio-to-Image Generator with Whisper (Fine-tuned) & Stable Diffusion")

st.write("Record an audio input and generate an image based on the transcribed text.")

# Define recording duration
duration = st.slider("Select duration of recording (seconds):", 1, 10, 5)

# Function to apply stemming to the transcription
def apply_stemming(text):
    words = nltk.word_tokenize(text)
    stemmed_words = [stemmer.stem(word) for word in words]
    return ' '.join(stemmed_words)

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

    # Step 1: Transcribe audio with the fine-tuned Whisper model
    st.write("Transcribing audio with Whisper model...")
    audio_input, _ = librosa.load(audio_path, sr=16000)  # Ensure the audio is loaded at 16kHz
    input_features = processor(audio_input, sampling_rate=16000, return_tensors="pt").input_features.to(whisper_model.device)
    predicted_ids = whisper_model.generate(input_features)
    transcription = processor.decode(predicted_ids[0], skip_special_tokens=True)
    st.write(f"Transcription: {transcription}")

    # Step 2: Apply NLP stemming to the transcription
    st.write("Applying NLP stemming...")
    stemmed_transcription = apply_stemming(transcription)
    st.write(f"Stemmed Transcription: {stemmed_transcription}")

    # Step 3: Generate image based on stemmed transcription
    st.write("Generating image with Stable Diffusion...")
    generated_image = pipe(stemmed_transcription).images[0]

    # Display the generated image
    st.image(generated_image, caption="Generated Image")
