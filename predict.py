"""
Classify your own tweets / sentences with the trained model.

Usage:
    python predict.py "I can't believe how good this concert was!!"
    python predict.py            (interactive mode - type a tweet, press Enter, Ctrl+C to quit)

Prints the prediction of the trained TF-IDF model and, for comparison, VADER's score.
"""

import sys

import joblib
from nltk.sentiment import SentimentIntensityAnalyzer

from sentiment_analysis import LABELS, clean_tweet, ensure_nltk_data

ensure_nltk_data()
model = joblib.load("outputs/sentiment_model.joblib")
sia = SentimentIntensityAnalyzer()


def classify(text: str) -> str:
    ml = LABELS[model.predict([clean_tweet(text)])[0]]
    v = sia.polarity_scores(text)["compound"]
    return f"{ml.upper():8s} (VADER compound {v:+.2f})"


if len(sys.argv) > 1:
    text = " ".join(sys.argv[1:])
    print(f"{classify(text)}  <-  {text}")
else:
    print("Type a tweet and press Enter (Ctrl+C to quit).")
    try:
        while True:
            text = input("> ")
            if text.strip():
                print(classify(text))
    except KeyboardInterrupt:
        print("\nBye!")
