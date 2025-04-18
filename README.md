☕ Sentiment Analysis of DeKUT Coffee on Twitter (X)
📌 Overview
This project performs sentiment analysis on public tweets related to DeKUT Coffee—a premium coffee brand grown and roasted by Dedan Kimathi University of Technology (DeKUT). Using Natural Language Processing (NLP) techniques, the project classifies tweets into Positive, Negative, or Neutral sentiments to better understand public perception and engagement.

🎯 Objective
The goal is to:

Gauge public sentiment towards DeKUT Coffee on social media

Identify key themes, praise, and pain points from customers

Generate insights to improve marketing, quality, and customer outreach

🧠 What It Does
loads tweets mentioning DeKUT Coffee

Cleans and preprocesses tweet text (remove mentions, hashtags, emojis, etc.)

Uses a pretrained or custom-trained model to analyze sentiment

Visualizes trends using word clouds, pie charts, and timelines

🛠️ Tech Stack

Component	Tools / Libraries
Data Collection	Tweepy, SNScrape
Data Cleaning	Pandas, re (regex), NLTK
Sentiment Model	TextBlob BERT 
Visualization	Matplotlib, Seaborn, WordCloud, Plotly
App Deployment	Streamlit 
💡 Features
🔍 Real-time sentiment analysis of tweets mentioning "DeKUT Coffee"

📊 Dashboard showing positive, neutral, and negative tweet proportions

📅 Time-series tracking of sentiment over time

☁️ Word clouds of frequently used words in each sentiment category

🧪Output
Sentiment Distribution:

🟢 Positive: 75%

🟡 Neutral: 15%

🔴 Negative: 10%

Common Positive Words: "rich", "aroma", "smooth", "energy boost"

Common Negative Words: "bitter", "delay", "overpriced"

