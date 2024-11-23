import streamlit as st 
import spacy 
from textblob import TextBlob 

# Load Spacy Model
nlp = spacy.load("en_core_web_sm")

# Streamlit UI
st.title("NLP-Powered User Interface")
st.sidebar.header("NLP Options")

# Input Text
user_input = st.text_area("Enter text for analysis:", "Streamlit is an amazing tool for building UIs!")

show_tokenization = st.sidebar.checkbox("Show Tokenization")
show_ner = st.sidebar.checkbox("Show Named Entities")
show_sentiment = st.sidebar.checkbox("Show Sentiment Analysis")
show_pos = st.sidebar.checkbox("Show Part-of-Speech (POS) Tags")

# Process Text
if user_input:
    doc = nlp(user_input)
    blob = TextBlob(user_input)

    st.subheader("Original Text")
    st.write(user_input)

    # Tokenization
    if show_tokenization:
        st.subheader("Tokenization")
        tokens = [token.text for token in doc]
        st.write(tokens)

    # Named Entity Recognition (NER)
    if show_ner:
        st.subheader("Named Entity Recognition")
        if doc.ents:
            for ent in doc.ents:
                st.write(f"Entity: `{ent.text}` - Label: `{ent.label_}`")
        else:
            st.write("No named entities found.")

    # Sentiment Analysis
    if show_sentiment:
        st.subheader("Sentiment Analysis")
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        st.write(f"Polarity: `{polarity}` (Range: -1 to 1)")
        st.write(f"Subjectivity: `{subjectivity}` (Range: 0 to 1)")

    # Part-of-Speech (POS) Tagging
    if show_pos:
        st.subheader("Part-of-Speech (POS) Tagging")
        pos_tags = [(token.text, token.pos_) for token in doc]
        st.write(pos_tags)
