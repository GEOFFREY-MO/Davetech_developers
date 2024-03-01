import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.data.path.append('/path/to/nltk_data')  # Set NLTK data path

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

# Function to load dataset
def load_dataset(file_path):
    return pd.read_csv(file_path)

# Function for basic data exploration
def explore_data(dataset):
    st.write("First 5 rows of the dataset:")
    st.write(dataset.head())
    st.write("Last 5 rows of the dataset:")
    st.write(dataset.tail())
    st.write("Basic statistics of the dataset:")
    st.write(dataset.describe())

# Function to visualize sentiment distribution
def visualize_sentiment_distribution(dataset):
    sentiment_counts = dataset['Sentiment'].value_counts()
    fig, ax = plt.subplots(figsize=(8, 6))
    sentiment_counts.plot(kind='bar', color=['green', 'red', 'blue'], ax=ax)
    ax.set_title('Distribution of Sentiments')
    ax.set_xlabel('Sentiment')
    ax.set_ylabel('Count')
    ax.tick_params(axis='x', rotation=0)
    st.pyplot(fig)

# Function for data preprocessing
def preprocess_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words if word not in set(stopwords.words('english'))]
    text = ' '.join(words)
    return text

# Function for feature extraction using TF-IDF
def extract_features(X_train, X_test):
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    return X_train_tfidf, X_test_tfidf

# Function for making predictions
def predict_sentiment(user_input, model, tfidf_vectorizer):
    preprocessed_input = preprocess_text(user_input)
    vectorized_input = tfidf_vectorizer.transform([preprocessed_input])
    prediction = model.predict(vectorized_input)
    return prediction[0]

# Main function
def main():
    # Load dataset
    dataset = load_dataset('brookeside.csv')

    # Preprocess the dataset
    dataset['clean_tweets'] = dataset['tweets'].apply(preprocess_text)

    # Separate labeled and unlabeled tweets
    labeled_tweets = dataset[dataset['Sentiment'].notnull()]
    unlabeled_tweets = dataset[dataset['Sentiment'].isnull()]

    # Train the model on labeled tweets
    X_train, X_test, y_train, y_test = train_test_split(labeled_tweets['clean_tweets'], labeled_tweets['Sentiment'], test_size=0.2, random_state=42)
    X_train_tfidf, X_test_tfidf = extract_features(X_train, X_test)
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)
    
    # Initialize TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    tfidf_vectorizer.fit(labeled_tweets['clean_tweets'])

    # Sidebar options
    st.sidebar.title('Navigation')
    option = st.sidebar.selectbox('Go to', ['Home', 'Explore Data', 'Visualize Sentiment Distribution', 'Predict Sentiment'])

    # Home
    if option == 'Home':
        st.title('Sentiment Analysis App')

    # Explore Data
    elif option == 'Explore Data':
        st.title('Explore Data')
        explore_data(dataset)

    # Visualize Sentiment Distribution
    elif option == 'Visualize Sentiment Distribution':
        st.title('Visualize Sentiment Distribution')
        visualize_sentiment_distribution(dataset)

    # Prediction
    elif option == 'Predict Sentiment':
        st.title('Predict Sentiment')
        user_input = st.text_input("Enter a tweet:", "")
        if st.button("Predict"):
            prediction = predict_sentiment(user_input, model, tfidf_vectorizer)
            st.write("Predicted Sentiment:", prediction)

if __name__ == "__main__":
    main()
