import streamlit as st
import pandas as pd
from transformers import pipeline

# Load the sentiment analysis pipeline
pipe = pipeline("text-classification", model="anchit48/fine-tuned-sentiment-analysis-customer-feedback")

# Define a function to analyze sentiment and return emoji
def analyze_sentiment(text):
    result = pipe(text)
    sentiment = result[0]['label']

    # Map the sentiment to an emoji
    if sentiment == 'LABEL_2':  # Assuming 'LABEL_2' is Positive
        emoji = "😊"
    elif sentiment == 'LABEL_1':  # Assuming 'LABEL_1' is Neutral
        emoji = "😐"
    elif sentiment == 'LABEL_0':  # Assuming 'LABEL_0' is Negative
        emoji = "😞"
    else:
        emoji = "😊"  # Fallback emoji for unexpected labels

    return f"{sentiment} {emoji}"

# Streamlit UI elements
st.title("Sentiment Analysis with Soumi's Model")

st.write("""
Upload a CSV file with a 'Text' column or input your own text to perform sentiment analysis.
The model used is fine-tuned for customer feedback sentiment analysis.
""")

# User text input for single sentiment analysis
st.header("Analyze Sentiment of a Single Text")
user_input = st.text_input("Enter text for sentiment analysis")

if user_input:
    sentiment_result = analyze_sentiment(user_input)
    st.write(f"Sentiment: {sentiment_result}")

# File upload for batch sentiment analysis
st.header("Batch Sentiment Analysis via CSV")
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is not None:
    # Read the uploaded CSV file
    df = pd.read_csv(uploaded_file)
    
    # Check if the 'Text' column exists
    if 'Text' in df.columns:
        st.write("Data Preview:")
        st.write(df.head())
        
        # Apply sentiment analysis to each row in the DataFrame
        st.write("Performing sentiment analysis...")
        df['Sentiment'] = df['Text'].apply(analyze_sentiment)
        
        # Show the result in Streamlit
        st.write("Sentiment analysis results:")
        st.write(df.head())
        
        # Provide a download button for the updated CSV
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download results as CSV",
            data=csv,
            file_name='sentiment_analysis_results_with_emoji.csv',
            mime='text/csv',
        )
    else:
        st.error("The uploaded CSV does not contain a 'Text' column.")
else:
    st.write("Please upload a CSV file to perform batch sentiment analysis.")

