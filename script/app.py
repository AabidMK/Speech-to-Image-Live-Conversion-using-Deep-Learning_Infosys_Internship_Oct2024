import whisper
import torch
import sounddevice as sd
import streamlit as st
from scipy.io.wavfile import write
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Set page configuration
st.set_page_config(page_title="Audio-to-Image Generator", layout="wide")
nltk.download("vader_lexicon")
sentiment_analyzer = SentimentIntensityAnalyzer()

# Custom theme
st.markdown(
    """
    <style>
    body {
        background-color: #1a1f36;
        color: white;
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background-color: #1a1f36;
    }
    h1, h2, h3 {
        color: #ff4d8f;
    }
    .stButton > button {
        background: linear-gradient(135deg, #ff4d8f, #ff6b6b);
        border: none;
        border-radius: 12px;
        color: white;
        font-weight: bold;
        padding: 10px 20px;
    }
    .stButton > button:hover {
        background: #ff6b6b;
    }
    .stSlider > div {
        color: #ff4d8f;
    }
    .stMarkdown p {
        color: rgba(255, 255, 255, 0.8);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🎨 Audio-to-Image Generator with Sentiment Analysis")
st.markdown("Record your audio, analyze its sentiment, and generate images based on your transcription!")

# Load models
@st.cache_resource
def load_models():
    st.write("Loading models...")
    whisper_model = whisper.load_model("base", device="cuda")
    sd_model_id = "stabilityai/stable-diffusion-2-1"
    pipe = StableDiffusionPipeline.from_pretrained(sd_model_id, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")
    return whisper_model, pipe

whisper_model, pipe = load_models()

# Function to record audio
def record_audio(duration=5, samplerate=16000):
    st.info("🎤 Recording audio... Speak now!")
    audio = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, dtype="float32")
    sd.wait()
    st.success("Recording complete!")
    return audio

# Function to save audio
def save_audio(audio, samplerate=16000):
    audio_path = "recorded_audio.wav"
    write(audio_path, samplerate, (audio * 32767).astype("int16"))
    return audio_path

# Function to transcribe audio
def transcribe_audio(audio_path):
    st.info("📝 Transcribing audio...")
    try:
        result = whisper_model.transcribe(audio_path)
        return result["text"]
    except Exception as e:
        st.error(f"Error during transcription: {e}")
        return None

# Function to analyze sentiment
def analyze_sentiment(text):
    scores = sentiment_analyzer.polarity_scores(text)
    sentiment = "Positive" if scores["compound"] > 0.05 else "Negative" if scores["compound"] < -0.05 else "Neutral"
    return sentiment, scores

# Function to generate an image
def generate_image(prompt):
    st.info("🎨 Generating image...")
    try:
        image = pipe(prompt).images[0]
        return image
    except Exception as e:
        st.error(f"Error during image generation: {e}")
        return None

# Main logic
if st.button("🎤 Record & Generate Image"):
    # Record audio
    duration = st.slider("Select recording duration (seconds):", 3, 10, 5)
    audio = record_audio(duration)

    # Save the recorded audio
    audio_path = save_audio(audio)

    # Transcribe audio
    transcription = transcribe_audio(audio_path)
    if transcription:
        st.success("Audio transcribed successfully!")
        st.write(f"**Transcription:** {transcription}")

        # Analyze sentiment
        sentiment, scores = analyze_sentiment(transcription)
        st.markdown(f"**Sentiment:** {sentiment}")
        st.markdown(f"**Sentiment Scores:** {scores}")

        # Generate image if sentiment is not negative
        if sentiment == "Negative":
            st.warning("Negative sentiment detected. Skipping image generation.")
        else:
            st.write("Generating image based on transcription...")
            image = generate_image(transcription)
            if image:
                st.image(image, caption="Generated Image")
            else:
                st.error("Image generation failed. Please try again.")
    else:
        st.error("Transcription failed. Please try again.")

st.markdown("---")
st.write(""" © 2024 [Tej Prakash ](#). All rights reserved.
         👨‍💻 Powered by Whisper, Stable Diffusion, and Sentiment Analysis
         This application is developed as part of a final project integrating Whisper, Stable Diffusion, and Sentiment Analysis.  
         Unauthorized reproduction or distribution of the application or its components is prohibited.
""")

