# src/sentiment.py
from nltk.sentiment import SentimentIntensityAnalyzer

def analyze_sentiment_vader(text):
    sia = SentimentIntensityAnalyzer()
    return sia.polarity_scores(text)
