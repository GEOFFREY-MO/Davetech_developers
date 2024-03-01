import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Function to load dataset
def load_dataset(file_path):
  return pd.read_csv(file_path, encoding='latin1')

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
  text = re.sub('[^a-zA-Z]', ' ', str(text))  # Ensure text is converted to string
  text = text.lower()
  words = text.split()
  lemmatizer = WordNetLemmatizer()
  words = [lemmatizer.lemmatize(word) for word in words if word not in set(stopwords.words('english'))]
  text = ' '.join(words)
  return text

# Function for feature extraction using TF-IDF
def extract_features(X_train, X_test, tfidf_vectorizer):
  # No need to recreate vectorizer here, use the passed one
  X_train_tfidf = tfidf_vectorizer.transform(X_train)
  X_test_tfidf = tfidf_vectorizer.transform(X_test)
  return X_train_tfidf, X_test_tfidf

# Function for making predictions
def predict_sentiment(user_input, model, tfidf_vectorizer):
  preprocessed_input = preprocess_text(user_input)
  # Transform the preprocessed input using the same TF-IDF vectorizer used during training
  vectorized_input = tfidf_vectorizer.transform([preprocessed_input])
  prediction = model.predict(vectorized_input)[0]
  return prediction

# Main function
def main():
  # Error handling for file upload (optional)
  try:
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
      dataset = load_dataset(uploaded_file)

      # Preprocess the dataset
      dataset['clean_tweets'] = dataset['tweets'].apply(preprocess_text)

      # Separate labeled and unlabeled tweets
      labeled_tweets = dataset[dataset['Sentiment'].notnull()]
      unlabeled_tweets = dataset[dataset['Sentiment'].isnull()]

      # Train the model on labeled tweets
      X_train, X_test, y_train, y_test = train_test_split(labeled_tweets['clean_tweets'], labeled_tweets['Sentiment'], test_size=0.2, random_state=42)

      # Initialize TF-IDF vectorizer (once for all operations)
      tfidf_vectorizer = TfidfVectorizer(max_features=5000)
      tfidf_vectorizer.fit(labeled_tweets['clean_tweets'])

      X_train_tfidf, X_test_tfidf = extract_features(X_train, X_test, tfidf_vectorizer)
      model = MultinomialNB()
      model.fit(X_train_tfidf, y_train)

            # Sidebar options
      st.sidebar.title('Navigation')
      option = st.sidebar.selectbox('Go to', ['Home', 'Explore Data', 'Visualize Sentiment Distribution', 'Predict Sentiment'])

      # Home
      if option == 'Home':
        st.title('Sentiment Analysis Dekut Coffee Tweets App')

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
          try:
            prediction = predict_sentiment(user_input, model, tfidf_vectorizer)
            st.write("Predicted Sentiment:", prediction)
          except Exception as e:  # Handle potential errors during prediction
            st.error(f"An error occurred: {e}")

  except Exception as e:  # Handle errors during file upload or other parts
    st.error(f"An error occurred: {e}")

if __name__ == "__main__":
  main()

