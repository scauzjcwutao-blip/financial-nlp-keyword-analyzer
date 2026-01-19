# src/sentiment.py
"""
Sentiment analysis for English financial texts using VADER (Valence Aware Dictionary and sEntiment Reasoner).
"""

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from typing import Dict

# Ensure required NLTK data is available
try:
    _sia = SentimentIntensityAnalyzer()
except LookupError:
    # Download the VADER lexicon if not already present
    nltk.download('vader_lexicon', quiet=True)
    _sia = SentimentIntensityAnalyzer()

def analyze_sentiment_vader(text: str) -> Dict[str, float]:
    """
    Analyzes the sentiment of an English text using VADER.
    
    Args:
        text (str): Input English text (e.g., financial news, reports).
        
    Returns:
        Dict[str, float]: A dictionary with keys 'neg', 'neu', 'pos', and 'compound',
                          representing negative, neutral, positive, and normalized
                          compound sentiment scores respectively.
                          
    Example:
        >>> result = analyze_sentiment_vader("The stock market is booming!")
        >>> print(result['compound'])  # e.g., 0.6249
    """
    if not isinstance(text, str):
        raise ValueError("Input must be a string.")
    
    return _sia.polarity_scores(text)

