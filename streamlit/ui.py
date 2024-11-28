#!pip install streamlit
import os
import streamlit as st
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torchaudio # type: ignore

from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Specify the pre-trained model you want to use (e.g., "openai/whisper-small")
pretrained_model_name = "openai/whisper-small"

from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Correctly load the model and processor
pretrained_model_name = "openai/whisper-small"
processor = WhisperProcessor.from_pretrained(pretrained_model_name)
model = WhisperForConditionalGeneration.from_pretrained(pretrained_model_name)

print("Model type:", type(model))  

save_path = "/content/drive/MyDrive/Train"
model.save_pretrained(save_path)
processor.save_pretrained(save_path)

print("Model and processor saved successfully!")


st.title("Speech-to-Text with Fine-tuned Whisper Model")
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

if uploaded_file is not None:
    waveform, sample_rate = torchaudio.load(uploaded_file)
    input_features = processor(waveform, sampling_rate=sample_rate, return_tensors="pt").input_features

    # Generate transcription
    with torch.no_grad():
        predicted_ids = model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    # Display the transcription
    st.subheader("Transcription:")
    st.text(transcription)
