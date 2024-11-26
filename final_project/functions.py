# -*- coding: utf-8 -*-
"""functions.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1mOWEgVUSCI9kuxrNXvRR8Z2owR9uZ3IJ
"""

import sounddevice as sd
import torch
import numpy as np
from scipy.io.wavfile import write
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline
from diffusers import StableDiffusionPipeline
import librosa

# Paths to models
WHISPER_MODEL_PATH = "C:/Users/prash/whisper-finetuned-v2"
SD_MODEL_ID = "runwayml/stable-diffusion-v1-5"

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load models
def load_whisper_model():
    processor = WhisperProcessor.from_pretrained(WHISPER_MODEL_PATH)
    model = WhisperForConditionalGeneration.from_pretrained(WHISPER_MODEL_PATH).to(DEVICE)
    return processor, model

def load_stable_diffusion():
    pipe = StableDiffusionPipeline.from_pretrained(SD_MODEL_ID, torch_dtype=torch.float16).to(DEVICE)
    return pipe

def load_sentiment_analysis():
    return pipeline("sentiment-analysis", device=0 if torch.cuda.is_available() else -1)

# Audio recording function
def record_audio(duration, output_path="audio_input.wav", fs=16000):
    """
    Record audio for a specified duration and save to output_path.
    """
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="float32")
    sd.wait()  # Wait until recording is finished
    write(output_path, fs, (audio * 32767).astype(np.int16))
    return output_path

# Transcription using Whisper
def transcribe_audio(audio_path, processor, model):
    """
    Transcribe audio using Whisper model.
    """
    audio_input, _ = librosa.load(audio_path, sr=16000)
    input_features = processor(audio_input, sampling_rate=16000, return_tensors="pt").input_features.to(DEVICE)
    predicted_ids = model.generate(input_features)
    transcription = processor.decode(predicted_ids[0], skip_special_tokens=True)
    return transcription

# Sentiment analysis
def analyze_sentiment(text, sentiment_pipeline):
    """
    Perform sentiment analysis on text.
    """
    result = sentiment_pipeline(text)
    sentiment_label = result[0]["label"]
    sentiment_score = result[0]["score"]
    return sentiment_label, sentiment_score

# Image generation
def generate_image(text, pipe):
    """
    Generate an image from text using Stable Diffusion.
    """
    with torch.no_grad():
        image = pipe(text).images[0]
    return image