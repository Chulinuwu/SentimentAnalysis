import streamlit as st
from transformers import pipeline
import pandas as pd
import plotly.express as px

# ตั้งค่าโมเดลและ pipeline
@st.cache_resource
def load_pipeline():
    return pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

pipe = load_pipeline()

# UI สำหรับผู้ใช้งาน
st.title("Sentiment Analysis ")
st.write("Analyze the sentiment of your text and visualize the results.")

# Input text
user_input = st.text_area("Enter text for sentiment analysis:", value="I like you. I love you")

if st.button("Analyze"):
    if user_input.strip():
        # Predict
        results = pipe(user_input, return_all_scores=True)
        results = results[0]  # Extract only the first result for simplicity

        # Convert to DataFrame
        df = pd.DataFrame(results)
        df['label'] = df['label'].str.capitalize()  # Format labels

        # Show table
        st.subheader("Results")
        st.dataframe(df)

        # Plot bar chart
        fig = px.bar(df, x='label', y='score', title="Sentiment Scores", labels={"label": "Sentiment", "score": "Score"})
        st.plotly_chart(fig)
    else:
        st.warning("Please enter some text to analyze.")


st.write("Model : https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest")
