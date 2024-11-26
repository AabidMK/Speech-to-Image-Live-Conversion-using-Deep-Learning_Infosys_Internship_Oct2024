from flask import Flask, render_template, request, jsonify, send_file
from flai import ToxicityClassifier
from diffusers import StableDiffusionPipeline
import whisper
from PIL import Image
import torch
import os

app = Flask(__name__)

stable_diffusion = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to("cuda")

whisper_model = whisper.load_model("base")

toxicity_classifier = ToxicityClassifier()

def analyze_toxicity(prompt):
    """Analyze text for toxicity using flai."""
    result = toxicity_classifier.predict(prompt)
    toxic_score = result['toxicity']
    return {"label": "TOXIC" if toxic_score > 0.7 else "NOT TOXIC", "score": toxic_score}

def analyze_and_filter_prompt(prompt):
    """Analyze and filter text prompt for toxicity."""
    # Toxicity detection
    toxicity_result = analyze_toxicity(prompt)
    if toxicity_result["label"] == "TOXIC" and toxicity_result["score"] > 0.7:
        return {"error": "Toxic language detected."}
    return {"sentiment": "NEUTRAL", "confidence": 0.0}

@app.route("/")
def home():
    """Render the home page."""
    return render_template("index.html")

@app.route("/transcribe", methods=["POST"])
def transcribe():
    """Transcribe audio to text."""
    audio_file = request.files.get("file")
    if not audio_file:
        return jsonify({"error": "No audio file provided."}), 400
    
    file_path = "temp_audio.wav"
    audio_file.save(file_path)
    
    try:
        result = whisper_model.transcribe(file_path)
        os.remove(file_path)  
        return jsonify({"text": result.get("text", "")})
    except Exception as e:
        os.remove(file_path)  
        return jsonify({"error": str(e)}), 500

@app.route("/generate", methods=["POST"])
def generate():
    """Generate an image based on a text prompt."""
    data = request.json
    prompt = data.get("prompt", "")
    if not prompt:
        return jsonify({"error": "No prompt provided."}), 400

    # Analyze prompt for toxicity before proceeding with image generation
    toxicity_check = analyze_and_filter_prompt(prompt)
    if "error" in toxicity_check:
        return jsonify(toxicity_check), 400

    try:
        # Generate the image using Stable Diffusion
        image = stable_diffusion(prompt).images[0]
        image_path = "generated_image.png"
        image.save(image_path)  # Save the image to disk
        
        # Send the image back as a response
        return send_file(image_path, mimetype="image/png")
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
