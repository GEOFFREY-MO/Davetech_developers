import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from transformers import pipeline
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud

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

# Function to load sentiment analysis pipeline
def load_sentiment_pipeline():
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

# Function for making predictions using the sentiment analysis pipeline
def predict_sentiment(text, sentiment_pipeline):
    preprocessed_text = preprocess_text(text)
    result = sentiment_pipeline(preprocessed_text)[0]
    
    label_map = {
        "LABEL_0": "Negative",
        "LABEL_1": "Neutral",
        "LABEL_2": "Positive"
    }
    
    sentiment = label_map[result['label']]
    return sentiment

# Function to analyze the entire dataset and add sentiment labels
def analyze_dataset(dataset, sentiment_pipeline):
    dataset['Predicted_Sentiment'] = dataset['clean_tweets'].apply(lambda x: predict_sentiment(x, sentiment_pipeline))
    return dataset

# Function to generate word cloud
def generate_word_cloud(dataset, sentiment):
    words = ' '.join(dataset[dataset['Predicted_Sentiment'] == sentiment]['clean_tweets'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(words)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

# Main function
def main():
    # Error handling for file upload (optional)
    try:
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded_file is not None:
            dataset = load_dataset(uploaded_file)

            # Preprocess the dataset
            dataset['clean_tweets'] = dataset['tweets'].apply(preprocess_text)

            # Load sentiment analysis pipeline
            sentiment_pipeline = load_sentiment_pipeline()

            # Analyze the dataset
            dataset = analyze_dataset(dataset, sentiment_pipeline)

            # Sidebar options
            st.sidebar.title('Navigation')
            option = st.sidebar.selectbox('Go to', ['Home', 'Explore Data', 'Visualize Sentiment Distribution', 'Predict Sentiment', 'Word Cloud', 'Filter Tweets'])

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
                        prediction = predict_sentiment(user_input, sentiment_pipeline)
                        st.write("Predicted Sentiment:", prediction)
                    except Exception as e:  # Handle potential errors during prediction
                        st.error(f"An error occurred: {e}")

            # Word Cloud
            elif option == 'Word Cloud':
                st.title('Word Cloud')
                sentiment_option = st.selectbox('Select Sentiment', ['Positive', 'Neutral', 'Negative'])
                if st.button("Generate Word Cloud"):
                    generate_word_cloud(dataset, sentiment_option)

            # Filter Tweets
            elif option == 'Filter Tweets':
                st.title('Filter Tweets')
                sentiment_option = st.selectbox('Select Sentiment to Filter', ['Positive', 'Neutral', 'Negative'])
                filtered_tweets = dataset[dataset['Predicted_Sentiment'] == sentiment_option]
                st.write(filtered_tweets[['tweets', 'Predicted_Sentiment']])

    except Exception as e:  # Handle errors during file upload or other parts
        st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
