# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
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
    plt.figure(figsize=(8, 6))
    sentiment_counts.plot(kind='bar', color=['green', 'red', 'blue'])
    plt.title('Distribution of Sentiments')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    st.pyplot()

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

# Function for model training and evaluation
def train_evaluate_model(X_train, X_test, y_train, y_test):
    naive_bayes = MultinomialNB()
    naive_bayes.fit(X_train, y_train)
    y_pred = naive_bayes.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    return accuracy, report, confusion

# Main function
def main():
    # Load dataset
    dataset = load_dataset('brookeside.csv')

    # Sidebar options
    st.sidebar.title('Navigation')
    option = st.sidebar.selectbox('Go to', ['Home', 'Explore Data', 'Visualize Sentiment Distribution'])

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

if __name__ == "__main__":
    main()
