import os
import streamlit as st
from transformers import pipeline
import pandas as pd
import plotly.express as px

# Suppress AVX2 FMA warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ตั้งค่าโมเดลและ pipeline
@st.cache_resource
def load_pipeline():
    return pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment-latest", top_k=None)

pipe = load_pipeline()

# UI สำหรับผู้ใช้งาน
st.title("Sentiment Analysis ")
st.write("Analyze the sentiment of your text and visualize the results.")

# Input text
user_input = st.text_area("Enter text for sentiment analysis:", value="I like you. I love you")

if st.button("Analyze"):
    # Perform sentiment analysis
    results = pipe(user_input)
    # Process results
    labels = [result['label'] for result in results[0]]
    scores = [result['score'] for result in results[0]]
    df = pd.DataFrame({'label': labels, 'score': scores})

    # Show table
    st.subheader("Results")
    st.dataframe(df)

    # Plot bar chart
    fig = px.bar(df, x='label', y='score', title="Sentiment Scores", labels={"label": "Sentiment", "score": "Score"})
    st.plotly_chart(fig)
else:
    st.warning("Please enter some text to analyze.")

st.write("Model : https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest")