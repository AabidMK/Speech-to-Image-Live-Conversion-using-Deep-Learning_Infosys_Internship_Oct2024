import streamlit as st
import sounddevice as sd
import torch
import numpy as np
from scipy.io.wavfile import write
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline
from diffusers import StableDiffusionPipeline
import librosa
import os
# os.environ["HF_HOME"] = r"D:\window ssd c\hugging facce"

# Paths to models
sd_model_id = "CompVis/stable-diffusion-v1-4"
whisper_model_path = "openai/whisper-base"
processor = WhisperProcessor.from_pretrained(whisper_model_path)
model = WhisperForConditionalGeneration.from_pretrained(whisper_model_path)

# Load Whisper model and processor
processor = WhisperProcessor.from_pretrained(whisper_model_path)
model = WhisperForConditionalGeneration.from_pretrained(whisper_model_path).to("cuda" if torch.cuda.is_available() else "cpu")

# Load Stable Diffusion model
pipe = StableDiffusionPipeline.from_pretrained(sd_model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
pipe = StableDiffusionPipeline.from_pretrained(sd_model_id)

# Load Hugging Face Sentiment Analysis Pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

# Streamlit UI setup
st.title("Audio-to-Image Generator with Sentiment Analysis")
st.write("Record your audio, transcribe it, analyze sentiment, and generate an image from the transcription.")

# Record audio
if st.button("Record"):
    duration = 5  # Set recording duration in seconds
    fs = 16000  # Sampling rate (16 kHz is compatible with Whisper)

    # Inform user of recording
    st.write("Recording...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()  # Wait until recording is finished
    st.write("Recording complete")

    # Save recorded audio to a file
    audio_path = "audio_input.wav"
    write(audio_path, fs, (audio * 32767).astype(np.int16))  # Scale to int16 for Whisper

    # Transcribe audio with Whisper
    st.write("Transcribing audio...")
    audio_input, _ = librosa.load(audio_path, sr=16000)  # Load and resample to 16 kHz
    input_features = processor(audio_input, sampling_rate=16000, return_tensors="pt").input_features.to(model.device)
    predicted_ids = model.generate(input_features)
    transcription = processor.decode(predicted_ids[0], skip_special_tokens=True)
    st.write("Transcription:", transcription)

    # Analyze sentiment
    st.write("Analyzing sentiment...")
    sentiment_result = sentiment_pipeline(transcription)[0]
    st.write("Sentiment Analysis Result:")
    st.write(f"Label: {sentiment_result['label']}, Score: {sentiment_result['score']:.2f}")

    # Generate image with Stable Diffusion
    st.write("Generating image from text...")
    with torch.no_grad():
        image = pipe(transcription, num_inference_steps=20).images[0]
    st.image(image, caption="Generated Image", use_container_width=True)

# Instructions
st.write("Click 'Record' to transcribe audio, analyze sentiment, and generate an image.")