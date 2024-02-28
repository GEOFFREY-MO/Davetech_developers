# Import necessary libraries
import pandas as pd
import streamlit as st
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Load dataset
@st.cache
def load_data():
    return pd.read_csv('brookeside.csv')

dataset = load_data()

# Model training
def train_model(dataset):
    # Preprocess the text data
    def preprocess_text(text):
        text = re.sub('[^a-zA-Z]', ' ', text)
        text = text.lower()
        words = text.split()
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words if word not in set(stopwords.words('english'))]
        return ' '.join(words)

    dataset['clean_tweets'] = dataset['tweets'].apply(preprocess_text)

    # TF-IDF Vectorization
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    X_tfidf = tfidf_vectorizer.fit_transform(dataset['clean_tweets'])
    y = dataset['Sentiment']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

    # Train Multinomial Naive Bayes classifier
    naive_bayes = MultinomialNB()
    naive_bayes.fit(X_train, y_train)

    return naive_bayes, tfidf_vectorizer

# Check if the model exists, if not, train it
model_path = "sentiment_model.pkl"
if not os.path.exists(model_path):
    model, tfidf_vectorizer = train_model(dataset)
    joblib.dump((model, tfidf_vectorizer), model_path)
else:
    model, tfidf_vectorizer = joblib.load(model_path)

# Streamlit app
def main():
    st.title("Sentiment Analysis App")

    st.sidebar.subheader("Model Evaluation")
    evaluate_model = st.sidebar.checkbox("Evaluate Model")

    if evaluate_model:
        # Model evaluation
        st.subheader("Model Evaluation")
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        classification_rep = classification_report(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)

        st.write("Accuracy:", accuracy)
        st.write("\nClassification Report:")
        st.write(classification_rep)
        st.write("\nConfusion Matrix:")
        st.write(conf_matrix)

    st.sidebar.subheader("Predict Sentiment")
    predict_sentiment = st.sidebar.checkbox("Predict Sentiment")

    if predict_sentiment:
        # Sentiment prediction
        st.subheader("Predict Sentiment")
        text_input = st.text_area("Enter text to predict sentiment:")
        if text_input:
            cleaned_text = preprocess_text(text_input)
            vectorized_text = tfidf_vectorizer.transform([cleaned_text])
            prediction = model.predict(vectorized_text)[0]
            st.write("Predicted Sentiment:", prediction)

if __name__ == "__main__":
    main()
