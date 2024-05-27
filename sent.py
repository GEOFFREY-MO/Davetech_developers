import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from transformers import pipeline
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
import nltk

# Ensure NLTK data is downloaded outside of Streamlit execution
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

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
  sentiment_counts = dataset['Predicted_Sentiment'].value_counts()
  fig, ax = plt.subplots(figsize=(8, 6))
  sentiment_counts.plot(kind='bar', color=['green', 'red', 'blue'], ax=ax)
  ax.set_title('Distribution of Sentiments')
  ax.set_xlabel('Sentiment')
  ax.set_ylabel('Count')
  ax.tick_params(axis='x', rotation=0)
  st.pyplot(fig)

# Function for data preprocessing
def preprocess_text(text, max_length=128):
  text = re.sub(r'http\S+', '', text)  # Remove URLs
  text = re.sub(r'@\w+', '', text)  # Remove user mentions
  text = re.sub(r'#', '', text)  # Remove hashtag symbol
  text = re.sub('[^a-zA-Z]', ' ', str(text))  # Remove punctuation and numbers
  text = text.lower()  # Convert to lowercase
  words = text.split()
  lemmatizer = WordNetLemmatizer()
  words = [lemmatizer.lemmatize(word) for word in words if word not in set(stopwords.words('english'))]
  text = ' '.join(words)
  return ' '.join(text.split()[:max_length])  # Truncate text to max_length words

# Function to load sentiment analysis pipeline
@st.cache(allow_output_mutation=True)
def load_sentiment_pipeline():
  return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

# Function for making predictions using the sentiment analysis pipeline
def predict_sentiment(text, sentiment_pipeline, max_length=128):
  preprocessed_text = preprocess_text(text, max_length=max_length)
  try:
    result = sentiment_pipeline(preprocessed_text)[0]
  except Exception as e:
    st.error(f"Error in prediction: {e}")
    return None

  label_map = {
    "LABEL_0": "Negative",
    "LABEL_1": "Neutral",
    "LABEL_2": "Positive"
  }

  sentiment = label_map.get(result['label'], "Unknown")
  return sentiment

# Function to analyze the entire dataset and add sentiment labels
def analyze_dataset(dataset, sentiment_pipeline, text_column, max_length=128):
  dataset['Predicted_Sentiment'] = dataset[text_column].apply(lambda x: predict_sentiment(x, sentiment_pipeline, max_length=max_length))
  return dataset

# Function to generate word cloud
def generate_word_cloud(dataset, sentiment):
  words = ' '.join(dataset[dataset['Predicted_Sentiment'] == sentiment]['clean_tweets'])
  wordcloud = WordCloud(width=800, height=400, background_color='white').generate(words)
  plt.figure(figsize=(10, 5))
  plt.imshow(wordcloud, interpolation='bilinear')
  plt.axis('off')
  st.pyplot(plt)

# Function to clean raw Twitter data
def clean_raw_twitter_data(text):
  text = re.sub(r'http\S+', '', text)  # Remove URLs
  text = re.sub(r'@\w+', '', text)  # Remove user mentions
  text = re.sub(r'#', '', text)  # Remove hashtag symbol
  text = re.sub('[^a-zA-Z]', ' ', str(text))  # Remove punctuation and numbers
  text = text.lower()  # Convert to lowercase
  words = text.split()
  lemmatizer = WordNetLemmatizer()
  words = [lemmatizer.lemmatize(word) for word in words if word not in set(stopwords.words('english'))]
  cleaned_text = ' '.join(words)
  return cleaned_text

# Main function
def main():
  # Create separate section outside sidebar for navigation
  navigation_section = st.container()
  option = navigation_section.selectbox('Go to', ['Home', 'Upload Dataset', 'How to Use', 'Clean Raw Tweets', 'Analyze Sentiment', 'Explore Data', 'Visualize Sentiment Distribution', 'Predict Sentiment', 'Word Cloud', 'Filter Tweets'])

  if option == 'Home':
    st.title('Sentiment Analysis Dekut Coffee Tweets App')
    st.write("Welcome to the Sentiment Analysis App for Dekut Coffee Tweets. Please upload a dataset to get started.")

  elif option == 'Upload Dataset':
    st.title('Upload Dataset')
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
      dataset = load_dataset(uploaded_file)
      st.session_state['dataset'] = dataset
      st.session_state['text_column'] = st.selectbox('Select the column containing the tweet text:', dataset.columns)

  elif option == 'How to Use':
    st.title('How to Use')
    st.write("""
    ### Step-by-Step User Guide
     
    **Step 1: Home**
    - When you first open the app, you will land on the **Home** page.
    - This page provides a welcome message and a brief introduction to the app.
     
    **Step 2: Upload Dataset**
    - Navigate to the **Upload Dataset** tab using the navigation section.
    - Click on the "Browse files" button to upload your CSV file containing the tweet data.
    - After uploading, you will be prompted to select the column that contains the tweet text. Choose the appropriate column from the dropdown menu.
     
    **Step 3: Clean Raw Tweets**
    - Once the dataset is uploaded and the text column is selected, go to the **Clean Raw Tweets** tab.
    - Click the "Clean Tweets" button. This will clean the raw tweets in the selected column and create a new column `clean_tweets` in the dataset.
     
    **Step 4: Analyze Sentiment**
    - Navigate to the **Analyze Sentiment** tab.
    - Click the "Analyze Sentiment" button. This will run sentiment analysis on the cleaned tweets and add a new column `Predicted_Sentiment` to the dataset.
     
    **Step 5: Explore Data**
    - Go to the **Explore Data** tab.
    - This section allows you to view the first and last five rows of the dataset, as well as basic statistics.
     
    **Step 6: Visualize Sentiment Distribution**
    - Navigate to the **Visualize Sentiment Distribution** tab.
    - This section provides a bar chart showing the distribution of sentiments (Positive, Neutral, Negative) in your dataset.
     
    **Step 7: Predict Sentiment**
    - Go to the **Predict Sentiment** tab.
    - Enter a tweet in the text input box and click "Predict" to see the predicted sentiment for that tweet.
     
    **Step 8: Generate Word Cloud**
    - Navigate to the **Word Cloud** tab.
    - Select the sentiment (Positive, Neutral, Negative) for which you want to generate a word cloud.
    - Click the "Generate Word Cloud" button to visualize the most common words for the selected sentiment.
     
    **Step 9: Filter Tweets**
    - Go to the **Filter Tweets** tab.
    - Select the sentiment (Positive, Neutral, Negative) to filter the tweets.
    - The filtered tweets will be displayed, showing only the tweets that match the selected sentiment.
     
    ### Notes
  - **Session State**: The app uses session state to track the progress of cleaning and sentiment analysis. Ensure you follow the steps in the given order to avoid any errors.
  - **Error Handling**: If you encounter any errors, such as missing columns or issues during prediction, appropriate error messages will be displayed.
  """)

  if 'dataset' in st.session_state and 'text_column' in st.session_state:
    dataset = st.session_state['dataset']
    text_column = st.session_state['text_column']

    if option == 'Clean Raw Tweets':
      st.title('Clean Raw Tweets')
      st.write("This function will clean the raw tweets and update the dataset with cleaned tweets.")
      if st.button("Clean Tweets"):
        st.write("Cleaning raw tweets...")
        dataset['clean_tweets'] = dataset[text_column].apply(clean_raw_twitter_data)
        st.session_state['cleaned'] = True
        st.write("Raw tweets cleaned successfully!")

    if 'cleaned' in st.session_state and st.session_state['cleaned']:
      if option == 'Analyze Sentiment':
        st.title('Analyze Sentiment')
        if st.button("Analyze Sentiment"):
          st.write("Analyzing sentiment...")
          sentiment_pipeline = load_sentiment_pipeline()
          dataset = analyze_dataset(dataset, sentiment_pipeline, 'clean_tweets')
          st.session_state['analyzed'] = True
          st.write("Sentiment analysis completed!")

      if 'analyzed' in st.session_state and st.session_state['analyzed']:
        if option == 'Explore Data':
          st.title('Explore Data')
          explore_data(dataset)

        if option == 'Visualize Sentiment Distribution':
          st.title('Visualize Sentiment Distribution')
          visualize_sentiment_distribution(dataset)

        if option == 'Predict Sentiment':
          st.title('Predict Sentiment')
          user_input = st.text_input("Enter a tweet:", "")
          if st.button("Predict"):
            sentiment_pipeline = load_sentiment_pipeline()
            prediction = predict_sentiment(user_input, sentiment_pipeline)
            if prediction:
              st.write("Predicted Sentiment:", prediction)

        if option == 'Word Cloud':
          st.title('Word Cloud')
          sentiment_option = st.selectbox('Select Sentiment', ['Positive', 'Neutral', 'Negative'])
          if st.button("Generate Word Cloud"):
            generate_word_cloud(dataset, sentiment_option)

        if option == 'Filter Tweets':
          st.title('Filter Tweets')
          sentiment_option = st.selectbox('Select Sentiment to Filter', ['Positive', 'Neutral', 'Negative'])
          filtered_tweets = dataset[dataset['Predicted_Sentiment'] == sentiment_option]
          st.write(filtered_tweets[[text_column, 'Predicted_Sentiment']])

  else:
    st.warning("Please upload a dataset and select the text column first.")

if __name__ == "__main__":
  main()
