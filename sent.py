import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
import os
import threading

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Define text preprocessing function
def preprocess_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words if word not in set(stopwords.words('english'))]
    text = ' '.join(words)
    return text

# Load the trained model
@st.cache
def load_model():
    model_path = "model.pkl"  # Path to your trained model
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        st.error("Model not found. Please train the model first.")

# Background function for preprocessing data
def preprocess_data(df):
    if 'tweets' in df.columns:
        df['clean_tweets'] = df['tweets'].apply(preprocess_text)
        return df
    else:
        st.error("Invalid CSV file. 'tweets' column not found.")
        return None

# Main function for Streamlit app
def main():
    st.title("Sentiment Analysis Web App")

    # Load model
    model = load_model()

    # Upload CSV file
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write(df.head())
        st.write("CSV file uploaded successfully!")

        # Start background thread for preprocessing data
        thread = threading.Thread(target=preprocess_data, args=(df,))
        thread.start()

    # Chat interface
    st.subheader("Chat Interface")
    message = st.text_input("Enter your message:")
    if st.button("Predict"):
        if not model:
            st.error("Model not found. Please train the model first.")
        else:
            cleaned_message = preprocess_text(message)
            prediction = model.predict([cleaned_message])
            st.write(prediction)

if __name__ == "__main__":
    main()
