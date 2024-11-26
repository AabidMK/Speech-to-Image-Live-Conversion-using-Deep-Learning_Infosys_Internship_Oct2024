# Speech-to-Image-Live-Conversion-using-Deep-Learning_Infosys_Internship_Oct2024
The objective of this project is to develop a deep learning model that can convert spoken descriptions into corresponding images in real-time.
# Speech-to-Image-Live-Conversion-using-Deep-Learning_Infosys_Internship_Oct2024
The objective of this project is to develop a deep learning model that can convert spoken descriptions into corresponding images in real-time.
This is a Speech-to-Image application that allows users to transcribe speech into text and then generate an image based on that text using Stable Diffusion. The app also analyzes the transcription for toxicity before generating the image.

## Requirements

- Python 3.7+
- Flask
- `whisper` for audio transcription
- `flai` for toxicity analysis
- `diffusers` for Stable Diffusion image generation
- `torch` for running Stable Diffusion on CUDA
- `PIL` for image handling

## Installation

1. Clone the repository or download the project files.

   ```bash
   git clone <repository_url>
   cd speech-to-image
   ```

2. Create a virtual environment (optional but recommended):

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use venv\Scripts\activate
   ```

3. Install required Python libraries:

   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

1. Make sure you have the required dependencies installed (Flask, Whisper, diffusers, etc.).

2. Run the Flask application:

   ```bash
   python app.py
   ```

   The application will be available at `http://localhost:5000`.

3. Open your web browser and navigate to the application. You'll see a button to record your voice.

4. When you press the microphone button, it will start recording your speech. After you stop recording, the speech will be transcribed into text, and an image based on that text will be generated and displayed on the page.

## How It Works

1. **Speech Recording:**
   - The frontend captures audio using the browser's `getUserMedia` API.
   - When the user presses the "🎤" button, recording starts. After stopping, the audio is sent to the backend.

2. **Transcription:**
   - The backend uses the Whisper model to transcribe the recorded audio into text.
   - The transcribed text is returned to the frontend.

3. **Sentiment and Toxicity Analysis:**
   - Before generating the image, the text is analyzed for toxicity using the `flai` Toxicity Classifier.
   - If the text is found to be toxic, the request is rejected.

4. **Image Generation:**
   - If the text is safe, it is sent to the Stable Diffusion model to generate an image based on the prompt.
   - The generated image is sent back to the frontend and displayed.

## Flowchart of the App

```plaintext
+---------------------+       +-----------------------+
| User Clicks Record  | ----> | Record Audio          |
| Button (🎤)         |       | (Start Recording)     |
+---------------------+       +-----------------------+
              |
              v
+---------------------+       +-----------------------+
| Audio Recorded      | ----> | Transcribe Audio to   |
| and Stopped         |       | Text using Whisper    |
+---------------------+       +-----------------------+
              |
              v
+---------------------+       +-----------------------+
| Transcription       | ----> | Sentiment Analyze Text for      |
| Returned to Frontend|       | Toxicity (Flai)       |
+---------------------+       +-----------------------+
              |
              v
+---------------------+       +-----------------------+
| Text is Safe        | ----> | Generate Image using  |
| (Non-toxic)         |       | Stable Diffusion      |
+---------------------+       +-----------------------+
              |
              v
+---------------------+       +-----------------------+
| Image Generated     | ----> | Display Image         |
| and Sent Back       |       | on Frontend           |
+---------------------+       +-----------------------+
```




