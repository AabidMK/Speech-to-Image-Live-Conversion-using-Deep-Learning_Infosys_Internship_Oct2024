import streamlit as st
import sounddevice as sd
import torch
import numpy as np
from scipy.io.wavfile import write
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline
from diffusers import StableDiffusionPipeline
import librosa
from time import sleep

# Paths to models
WHISPER_MODEL_PATH = r"C:\Users\Sabarinathan S\Desktop\streamlit\Speech-to-Image-Live-Conversion-using-Deep-Learning_Infosys_Internship_Oct2024-main\Speech-to-Image-Live-Conversion-using-Deep-Learning_Infosys_Internship_Oct2024-main\models\Whisper_finetuned"
SD_MODEL_ID = "CompVis/stable-diffusion-v1-4"

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load models
@st.cache_resource
def load_whisper_model():
    processor = WhisperProcessor.from_pretrained(WHISPER_MODEL_PATH)
    model = WhisperForConditionalGeneration.from_pretrained(WHISPER_MODEL_PATH).to(DEVICE)
    return processor, model

@st.cache_resource
def load_stable_diffusion():
    return StableDiffusionPipeline.from_pretrained(SD_MODEL_ID, torch_dtype=torch.float16).to(DEVICE)

@st.cache_resource
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

# Streamlit UI setup
st.set_page_config(page_title="Speech-to-Image Generator", layout="wide")

# Title and description
st.title("Speech-to-Image Generator")
st.write("Record your audio, transcribe it, perform sentiment analysis, and generate an image.")

# Load models
processor, model = load_whisper_model()
pipe = load_stable_diffusion()
sentiment_pipeline = load_sentiment_analysis()

# Audio recording duration slider
duration = st.slider(
    "Select recording duration (seconds)", 1, 30, 15, 1, help="Set the duration for recording."
)

# Record audio button
if st.button("Start Recording 🎙️"):
    st.info("Recording... Please speak clearly.")
    audio_path = record_audio(duration)
    st.success("Recording complete!")

    # Transcription
    st.write("**Transcribing audio...**")
    transcription = transcribe_audio(audio_path, processor, model)
    st.write(f"**Transcription:** {transcription}")

    # Sentiment analysis
    st.write("**Analyzing sentiment...**")
    sentiment_label, sentiment_score = analyze_sentiment(transcription, sentiment_pipeline)
    st.write(f"**Sentiment:** {sentiment_label} (Confidence: {sentiment_score:.2f})")

    # Image generation
    if sentiment_label == "NEGATIVE":
        st.warning("The sentiment is negative. No image will be generated.")
    else:
        st.write("**Generating image from transcription...**")
        with st.spinner("This may take a few seconds..."):
            sleep(2)
            image = generate_image(transcription, pipe)
        st.image(image, caption="Generated Image", use_column_width=True)  
