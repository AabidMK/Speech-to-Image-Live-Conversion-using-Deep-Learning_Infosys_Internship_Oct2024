import streamlit as st
import sounddevice as sd
import torch
import numpy as np
from scipy.io.wavfile import write
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from diffusers import StableDiffusionPipeline
import librosa

# Paths to models
whisper_model_path = "C:/Users/HP/OneDrive/Desktop/Infosys Springboard/whisper-finetuned" 
sd_model_id = "stabilityai/stable-diffusion-2-1"

# Load Whisper model and processor
processor = WhisperProcessor.from_pretrained(whisper_model_path)
model = WhisperForConditionalGeneration.from_pretrained(whisper_model_path).to("cuda" if torch.cuda.is_available() else "cpu")

# Load Stable Diffusion model
pipe = StableDiffusionPipeline.from_pretrained(sd_model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

# Streamlit UI setup
st.title("Audio-to-Image Generator")
st.write("Record your audio, transcribe it, and generate an image from the transcription.")

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

    # Generate image with Stable Diffusion
    st.write("Generating image from text...")
    with torch.no_grad():
        image = pipe(transcription).images[0]
    st.image(image, caption="Generated Image", use_column_width=True)

# Instructions
st.write("Click 'Record' to transcribe audio and generate an image.")
