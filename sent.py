import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import logging

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Add logging configuration
logging.basicConfig(level=logging.INFO)

# Load the trained model
@st.cache
def load_model():
    # Load your trained model here
    pass

# Define text preprocessing function
def preprocess_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words if word not in set(stopwords.words('english'))]
    text = ' '.join(words)
    return text

# Load the CSV file and preprocess the text
def preprocess_csv(file):
    df = pd.read_csv(file)
    df['clean_tweets'] = df['tweets'].apply(preprocess_text)
    return df['clean_tweets']

# Main function for Streamlit app
def main():
    st.title("Brookside Tweets analysis")

    # Load model
    model = load_model()

    # Upload CSV file
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write(df.head())
        st.write("CSV file uploaded successfully!")

        # Preprocess and predict sentiment
        X = preprocess_csv(uploaded_file)
        logging.info("Preprocessed input data: %s", X)  # Log preprocessed input data
        prediction = model.predict(X)
        st.write(prediction)

    # Chat interface
    st.subheader("Chat Interface")
    message = st.text_input("Enter your message:")
    if st.button("Predict"):
        cleaned_message = preprocess_text(message)
        logging.info("Cleaned message: %s", cleaned_message)  # Log cleaned message
        prediction = model.predict([cleaned_message])
        st.write(prediction)

if __name__ == "__main__":
    main()
