import streamlit as st
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import sounddevice as sd
from scipy.io.wavfile import write
import librosa
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Set page configuration at the very top
st.set_page_config(
    page_title="AI Audio-to-Image Generator",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Download necessary nltk data
nltk.download("vader_lexicon")
nltk.download("punkt")

# Initialize NLP tools
sentiment_analyzer = SentimentIntensityAnalyzer()

# Load Whisper and Stable Diffusion models
@st.cache_resource
def load_models():
    processor = WhisperProcessor.from_pretrained("C:/Users/HP/OneDrive/Desktop/Infosys Springboard/whisper-finetuned")
    whisper_model = WhisperForConditionalGeneration.from_pretrained("C:/Users/HP/OneDrive/Desktop/Infosys Springboard/whisper-finetuned")
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    return whisper_model, processor, pipe

whisper_model, processor, pipe = load_models()

# Custom theme initialization
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }

    :root {
        --primary-pink: #FF4D8F;
        --primary-blue: #2A93D5;
        --dark-bg: #0F1123;
        --card-bg: #1A1F37;
        --glow: 0 0 15px rgba(255, 77, 143, 0.2);
    }

    .stApp {
        background-color: var(--dark-bg);
        background-image: 
            radial-gradient(circle at 10% 20%, rgba(255, 77, 143, 0.1) 0%, transparent 20%),
            radial-gradient(circle at 90% 80%, rgba(42, 147, 213, 0.1) 0%, transparent 20%);
    }

    /* Card Styling */
    .stCard {
        background: var(--card-bg);
        border-radius: 16px;
        box-shadow: var(--glow);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
    }

    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-pink), #FF6B6B);
        border: none;
        border-radius: 12px;
        color: white;
        font-weight: 500;
        padding: 12px 24px;
        box-shadow: var(--glow);
        transition: transform 0.2s;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
    }

    /* Upload Area */
    .uploadfile {
        background: var(--card-bg);
        border: 2px dashed rgba(255, 255, 255, 0.2);
        border-radius: 16px;
        padding: 30px;
    }

    /* Slider Styling */
    .stSlider > div {
        color: var(--primary-pink);
    }

    .stSlider .stSlideHandle {
        background: var(--primary-pink);
    }

    /* Text Elements */
    h1, h2, h3 {
        color: white;
        font-weight: 600;
    }

    p {
        color: rgba(255, 255, 255, 0.8);
    }

    /* Add floating elements animation */
    .floating-element {
        animation: float 6s ease-in-out infinite;
    }

    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-20px); }
        100% { transform: translateY(0px); }
    }
    </style>
""", unsafe_allow_html=True)

# Layout structure
col1, col2 = st.columns([1, 3])

with col1:
    st.markdown(""" 
        <div class="stCard floating-element">
            <h3>Settings ⚙️</h3>
        </div>
    """, unsafe_allow_html=True)

    # Settings controls
    duration = st.slider("Recording Duration (seconds):", 1, 10, 5)
    sentiment_filter = st.checkbox("Enable Sentiment Filtering", value=True)

with col2:
    st.markdown("""
        <div class="stCard">
            <h1>🎙️ AI Audio-to-Image Generator</h1>
            <p>Transform audio into creative images and analyze emotions with AI! 🎨</p>
        </div>
    """, unsafe_allow_html=True)

# Core Functions
def analyze_sentiment(text):
    scores = sentiment_analyzer.polarity_scores(text)
    sentiment = "Positive" if scores["compound"] > 0.05 else "Negative" if scores["compound"] < -0.05 else "Neutral"
    return sentiment, scores

def generate_image(prompt):
    st.write("Generating image... Please wait ⏳")
    with torch.no_grad():
        image = pipe(prompt).images[0]
    return image

# Recording and Image Generation
col1, col2 = st.columns(2)
audio_path = None

with col1:
    if st.button("🎤 Record Audio"):
        st.info("Recording...")
        fs = 44100
        audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
        sd.wait()
        audio_path = "recorded_audio.wav"
        write(audio_path, fs, audio_data)
        st.success("Recording Complete!")

with col2:
    uploaded_file = st.file_uploader("Upload Audio File", type=["wav", "mp3"])
    if uploaded_file:
        audio_path = "uploaded_audio.wav"
        with open(audio_path, "wb") as f:
            f.write(uploaded_file.read())
        st.success("File Uploaded!")

if audio_path:
    st.subheader("🔍 Processing Audio")
    audio_input, _ = librosa.load(audio_path, sr=16000)
    input_features = processor(audio_input, sampling_rate=16000, return_tensors="pt").input_features.to(whisper_model.device)

    st.write("Transcribing...")
    predicted_ids = whisper_model.generate(input_features)
    transcription = processor.decode(predicted_ids[0], skip_special_tokens=True)
    st.write(f"**Transcription**: {transcription}")

    st.write("Analyzing Sentiment...")
    sentiment, scores = analyze_sentiment(transcription)
    st.write(f"**Sentiment**: {sentiment}")

    # Show sentiment scores in a light color box
    st.markdown(
        f"""
        <div class="stCard">
            <h4>Sentiment Analysis Scores</h4>
            <p>{scores}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    if sentiment_filter and sentiment == "Negative":
        st.warning("Negative sentiment detected. Image generation skipped.")
    else:
        generated_image = generate_image(transcription)
        st.markdown(
            """
            <div class="stCard">
                <h3>Generated Image</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.image(generated_image, use_column_width=True)

# Copyright Notice
st.markdown(
    """
    ---
    © 2024 [Anisha Kumari](#). All rights reserved.  
    This application is developed as part of a final project integrating Whisper, Stable Diffusion, and Sentiment Analysis.  
    Unauthorized reproduction or distribution of the application or its components is prohibited.
    """
)
