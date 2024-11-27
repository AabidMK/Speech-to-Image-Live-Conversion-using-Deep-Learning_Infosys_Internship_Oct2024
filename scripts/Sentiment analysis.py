from transformers import pipeline

def analyze_sentiment(text):
    """Analyze sentiment of the text using Hugging Face pipeline."""
    sentiment_pipeline = pipeline("sentiment-analysis")
    sentiment = sentiment_pipeline(text)
    return sentiment[0]  # Returns label and score
