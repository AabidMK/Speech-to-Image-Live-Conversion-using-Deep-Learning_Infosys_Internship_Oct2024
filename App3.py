#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np
import sys
import librosa
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from diffusers import StableDiffusionPipeline

st.title("Speech to Image Generator")
st.write("Initializing models, please wait...")

model_path = r"C:\Users\sarve\whisper-finetuned1"
processor = WhisperProcessor.from_pretrained(model_path)
whisper_model = WhisperForConditionalGeneration.from_pretrained(model_path)

stable_diffusion_model = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
st.write("Models are loaded and ready.")

def record_audio(duration, fs=16000):
    st.info("Recording started...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype=np.int16)
    sd.wait()
    st.success("Recording stopped.")
    return np.squeeze(audio)

def save_wav(filename, audio, fs=16000):
    wav.write(filename, fs, audio)

st.write("Click 'Record' to start recording your audio. After processing, the generated image will be displayed below.")
if st.button("Record"):
    duration = 5  # seconds
    audio = record_audio(duration)
    audio_filename = "mic_input.wav"
    save_wav(audio_filename, audio)
    audio_path = "mic_input.wav"
    audio_array, _ = librosa.load(audio_path, sr=16000)  
    audio_input = processor(audio_array, return_tensors="pt", sampling_rate=16000)
    
    with torch.no_grad():
        generated_ids = whisper_model.generate(**audio_input)
    
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    st.write(f"Transcription: {transcription}")
    
    with st.spinner("Generating image..."):
        image = stable_diffusion_model(transcription).images[0]
    
    st.image(image, caption="Generated Image", use_column_width=True)
    image.save("output_image.png")
