# Import necessary libraries
import streamlit as st
import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from diffusers import StableDiffusionPipeline
import torch
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Download the VADER lexicon
nltk.download('vader_lexicon')  # Ensure the VADER lexicon is available

# Initialize sentiment analyzer
sentiment_analyzer = SentimentIntensityAnalyzer()

# Caching the model loading
@st.cache_resource  # Cache the models to avoid reloading them every time
def load_models():
    # Load Whisper model and processor
    whisper_model = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-small", cache_dir="./models"
    )
    whisper_processor = WhisperProcessor.from_pretrained(
        "openai/whisper-small", cache_dir="./models"
    )
    
    # Load Stable Diffusion model
    stable_diffusion_model = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4"
    )
    # Modify the scheduler to use fewer steps (faster but lower quality)
    stable_diffusion_model.scheduler.num_inference_steps = 25
    
    return whisper_model, whisper_processor, stable_diffusion_model

# Load the models when the app starts
st.title("Speech to Image Generator with Sentiment Analysis")
st.write("Initializing models, please wait...")

# Call the load_models function to load the models
whisper_model, whisper_processor, stable_diffusion_model = load_models()

st.write("Models are loaded and ready.")

# Function for sentiment analysis
def analyze_sentiment(text):
    scores = sentiment_analyzer.polarity_scores(text)
    sentiment = "Positive" if scores["compound"] > 0.05 else "Negative" if scores["compound"] < -0.05 else "Neutral"
    return sentiment, scores

# Function to check for general negative sentiment
def contains_negative_sentiment(text):
    sentiment, _ = analyze_sentiment(text)
    return sentiment == "Negative"

# Define functions for recording audio and saving to a .wav file
def record_audio(duration, fs=16000):
    st.info("Recording started...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype=np.int16)
    sd.wait()  # Wait until the recording is finished
    st.success("Recording stopped.")
    return np.squeeze(audio)

def save_wav(filename, audio, fs=16000):
    wav.write(filename, fs, audio)

# Streamlit app interface
st.write("Click 'Record' to start recording your audio. After processing, the generated image will be displayed below.")

# Set up a button to start the recording
if st.button("Record"):
    duration = 5  # seconds
    audio = record_audio(duration)
    audio_filename = "mic_input.wav"
    save_wav(audio_filename, audio)

    # Load the audio file and convert it to a NumPy array
    sampling_rate, audio_data = wav.read(audio_filename)

    # Ensure the audio data is in the correct format (float32)
    audio_data = audio_data.astype(np.float32) / np.iinfo(np.int16).max

    # Transcribe audio using the Whisper model
    audio_input = whisper_processor(audio_data, sampling_rate=sampling_rate, return_tensors="pt").input_features
    result = whisper_model.generate(audio_input)
    transcription = whisper_processor.decode(result[0], skip_special_tokens=True)
    st.write(f"Transcription: {transcription}")

    # Perform sentiment analysis on the transcription
    sentiment, sentiment_scores = analyze_sentiment(transcription)
    st.write(f"Sentiment: {sentiment}")

    # Check sentiment and block negative prompts
    if contains_negative_sentiment(transcription):
        st.warning("Negative sentiment detected. Image generation blocked.")
    else:
        # Generate image using Stable Diffusion based on the transcription
        with st.spinner("Generating image..."):
            image = stable_diffusion_model(prompt=transcription, num_inference_steps=20).images[0]

        # Display the generated image
        st.image(image, caption="Generated Image", use_column_width=True)
        image.save("output_image.png")
