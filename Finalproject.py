#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np
import librosa
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')
sid = SentimentIntensityAnalyzer()
st.set_page_config(page_title="Speech to Image Generator", layout="wide")
st.title("🎤 Speech to Image Generator")
st.sidebar.title("Settings")
duration = st.sidebar.slider("Recording Duration (seconds)", 1, 10, 5)
fs = 16000  
st.write("Initializing models, please wait...")
model_path = r"C:\Users\sarve\whisper-finetuned1"
processor = WhisperProcessor.from_pretrained(model_path)
whisper_model = WhisperForConditionalGeneration.from_pretrained(model_path)
stable_diffusion_model = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
stable_diffusion_model.scheduler = EulerDiscreteScheduler.from_config(stable_diffusion_model.scheduler.config)

st.write("Models are loaded and ready.")

def record_audio(duration, fs=16000):
    st.info("Recording started...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype=np.int16)
    sd.wait()
    st.success("Recording stopped.")
    return np.squeeze(audio)

def save_wav(filename, audio, fs=16000):
    wav.write(filename, fs, audio)

def analyze_sentiment(transcription):
    """Analyze sentiment using VADER and return sentiment scores."""
    sentiment_scores = sid.polarity_scores(transcription)
    return sentiment_scores

def is_safe_transcription(transcription):
    """Check if transcription contains inappropriate content or negative sentiment."""
    sentiment_scores = analyze_sentiment(transcription)
    if sentiment_scores['compound'] < 0.2:
        return False
    
    inappropriate_keywords = ['sex', 'porn', 'nudity', 'violence']
    if any(keyword in transcription.lower() for keyword in inappropriate_keywords):
        return False  
    
    return True  

st.write("Click 'Record' to start recording your audio. After processing, the generated image will be displayed below.")

if st.button("Record"):
    audio = record_audio(duration)
    audio_filename = "mic_input.wav"
    save_wav(audio_filename, audio)
    
    audio_array, _ = librosa.load(audio_filename, sr=fs)  
    audio_input = processor(audio_array, return_tensors="pt", sampling_rate=fs)
    
    with torch.no_grad():
        generated_ids = whisper_model.generate(**audio_input)
    
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    st.write(f"**Transcription:** {transcription}")
    
    sentiment_scores = analyze_sentiment(transcription)
    st.write(f"**Sentiment Scores:** {sentiment_scores}")
    
    if not is_safe_transcription(transcription):
        st.warning("The transcription contains inappropriate content or negative sentiment. Image generation aborted.")
    else:
        with st.spinner("Generating image..."):
            image = stable_diffusion_model(transcription, num_inference_steps=30).images[0]
        
        st.image(image, caption="Generated Image", use_column_width=True)
        image.save("output_image.png")
st.sidebar.markdown("---")
st.sidebar.subheader("About")
st.sidebar.write("This application uses speech recognition and image generation technologies to create images based on spoken input.")
