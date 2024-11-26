import streamlit as st
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import librosa
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download required nltk resources
nltk.download("vader_lexicon")

# Initialize sentiment analyzer
sentiment_analyzer = SentimentIntensityAnalyzer()

# Load Whisper and Stable Diffusion models
@st.cache_resource  # Cache models to prevent reloading each time
def load_models():
    processor = WhisperProcessor.from_pretrained("C:/Users/SANJANA/Downloads/whisper-finetuned_with_voice_dataset")
    whisper_model = WhisperForConditionalGeneration.from_pretrained("C:/Users/SANJANA/Downloads/whisper-finetuned_with_voice_dataset")
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    return whisper_model, processor, pipe

whisper_model, processor, pipe = load_models()

st.title("Speech to Image Live Conversion using Deep Learning")

st.write("Record an audio input, generate an image, and analyze sentiment based on the transcribed text.")

# Define recording duration
duration = st.slider("Select duration of recording (seconds):", 1, 10, 5)

# Function for sentiment analysis
def analyze_sentiment(text):
    scores = sentiment_analyzer.polarity_scores(text)
    sentiment = "Positive" if scores["compound"] > 0.05 else "Negative" if scores["compound"] < -0.05 else "Neutral"
    return sentiment, scores

# Function to check for general negative sentiment
def contains_negative_sentiment(text):
    sentiment, _ = analyze_sentiment(text)
    return sentiment == "Negative"

# Function to handle image generation based on sentiment
def generate_image_based_on_sentiment(transcription):
    # Step 4: Sentiment Filtering for Image Generation
    if contains_negative_sentiment(transcription):
        st.warning("Negative sentiment detected. Image generation disabled.")
        return  # Exit function and prevent image generation

    st.write("Generating image...")
    with torch.no_grad():
        generated_image = pipe(transcription).images[0]
        st.image(generated_image, caption="Generated Image")

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

    # Step 2: Perform sentiment analysis on the transcription
    st.write("Analyzing sentiment of the transcription...")
    sentiment, sentiment_scores = analyze_sentiment(transcription)
    st.write(f"Sentiment: {sentiment}")
    st.write("Sentiment Scores:", sentiment_scores)

    # Step 3: Check sentiment, decide on image generation
    generate_image_based_on_sentiment(transcription)
