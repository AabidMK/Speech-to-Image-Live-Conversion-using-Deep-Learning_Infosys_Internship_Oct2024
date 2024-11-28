import streamlit as st
import sounddevice as sd # type: ignore
import torch
import numpy as np
from scipy.io.wavfile import write
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline
from diffusers import StableDiffusionPipeline # type: ignore
import librosa

# Paths to models
whisper_model_path = r"C:/Users/Sabarinathan S/Desktop/streamlit/whisper-finetuned"
sd_model_id = "stabilityai/stable-diffusion-2-1"

# Check GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load models
@st.cache_resource
def load_models():
    # Load Whisper
    whisper_processor = WhisperProcessor.from_pretrained(whisper_model_path)
    whisper_model = WhisperForConditionalGeneration.from_pretrained(whisper_model_path).to(device)
    # Load Stable Diffusion
    sd_pipe = StableDiffusionPipeline.from_pretrained(sd_model_id, torch_dtype=torch.float16).to(device)
    # Load sentiment analysis
    sentiment_analyzer = pipeline("sentiment-analysis", device=0 if torch.cuda.is_available() else -1)
    return whisper_processor, whisper_model, sd_pipe, sentiment_analyzer

whisper_processor, whisper_model, sd_pipe, sentiment_analyzer = load_models()

# Streamlit UI
st.title("Audio-to-Image Generator")
st.write("Record your voice, transcribe it, analyze sentiment, and generate an image if sentiment is positive or neutral.")

duration = st.slider("Recording duration (seconds):", 1, 30, 10)

if st.button("Record and Process"):
    st.write("Recording...")
    fs = 16000  
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    st.write("Recording complete!")

    # Save audio to a file
    audio_path = "audio_input.wav"
    write(audio_path, fs, (audio * 32767).astype(np.int16))

    # Transcribe audio
    st.write("Transcribing...")
    audio_input, _ = librosa.load(audio_path, sr=16000)
    input_features = whisper_processor(audio_input, sampling_rate=16000, return_tensors="pt").input_features.to(device)
    predicted_ids = whisper_model.generate(input_features)
    transcription = whisper_processor.decode(predicted_ids[0], skip_special_tokens=True)
    st.write("Transcription:", transcription)

    # Sentiment analysis
    st.write("Analyzing sentiment...")
    sentiment = sentiment_analyzer(transcription)[0]
    sentiment_label = sentiment['label']
    sentiment_score = sentiment['score']
    st.write(f"Sentiment: {sentiment_label} ({sentiment_score:.2f})")

    # Image generation
    if sentiment_label in ["POSITIVE", "NEUTRAL"]:
        st.write("Generating image...")
        image = sd_pipe(transcription,num_inference_steps=20).images[0]
        st.image(image, caption="Generated Image", use_column_width=True)
    else:
        st.write("Sentiment is negative. No image generated.")