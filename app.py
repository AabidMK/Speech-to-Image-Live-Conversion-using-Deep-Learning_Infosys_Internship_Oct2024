import streamlit as st
import requests
import speech_recognition as sr
from PIL import Image
from io import BytesIO

# Backend URL
BACKEND_URL = "https://60f8-34-126-93-69.ngrok-free.app"  # Replace with your backend's deployed URL if not running locally

# Streamlit app title
st.title("Speech-to-Image Generator")
st.write("Upload your voice or type a prompt to generate AI-generated images!")

# Speech recognition
def transcribe_audio():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Listening for your speech...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
        try:
            transcription = recognizer.recognize_google(audio)
            st.write("Transcription:", transcription)
            return transcription
        except sr.UnknownValueError:
            st.error("Could not understand the audio.")
        except sr.RequestError:
            st.error("Speech recognition API is unavailable.")

# Option to choose input method
st.sidebar.header("Input Method")
input_method = st.sidebar.radio("Choose input type", ("Type Prompt", "Use Microphone"))

# Handle text input
if input_method == "Type Prompt":
    prompt = st.text_input("Enter your prompt for the image:")
    if st.button("Generate Image from Prompt"):
        if prompt:
            st.write("Sending prompt to backend...")
            response = requests.post(f"{BACKEND_URL}/generate_image", json={"text": prompt})
            if response.status_code == 200:
                image_data = response.content
                image = Image.open(BytesIO(image_data))
                st.image(image, caption="Generated Image")
            else:
                st.error(f"Error: {response.json().get('error', 'Unknown error')}")

# Handle audio input
if input_method == "Use Microphone":
    if st.button("Record and Generate Image"):
        transcription = transcribe_audio()
        if transcription:
            st.write("Sending transcription to backend...")
            response = requests.post(f"{BACKEND_URL}/generate_image", json={"text": transcription})
            if response.status_code == 200:
                image_data = response.content
                image = Image.open(BytesIO(image_data))
                st.image(image, caption="Generated Image")
            else:
                st.error(f"Error: {response.json().get('error', 'Unknown error')}")

# Footer
st.write("Powered by FastAPI, Stable Diffusion, and Streamlit.")
