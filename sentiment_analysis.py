"""
Twitter Sentiment Analysis
==========================
Classifies tweets as positive / neutral / negative.

Two approaches are compared on the TweetEval sentiment benchmark
(SemEval-2017 Task 4A - 45k training / 12k test tweets):

  1. VADER            - NLTK's rule-based lexicon model (no training)
  2. TF-IDF + ML      - NLTK preprocessing, then Logistic Regression and Linear SVM

Run:
    python sentiment_analysis.py

Outputs (written to ./outputs):
    class_distribution.png   - tweets per sentiment class
    tweet_length.png         - tweet length by class
    top_words.png            - most frequent words per class
    vader_scores.png         - VADER compound score distribution per true class
    model_comparison.png     - accuracy / macro-F1 for each approach
    confusion_matrix.png     - confusion matrix of the best model
    sentiment_model.joblib   - trained pipeline (vectorizer + classifier)
    metrics.txt              - classification reports
"""

import os
import re
from collections import Counter

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

DATA_DIR = "data"
OUT_DIR = "outputs"
LABELS = ["negative", "neutral", "positive"]
os.makedirs(OUT_DIR, exist_ok=True)
sns.set_theme(style="whitegrid")
PALETTE = {"negative": "#d62728", "neutral": "#7f7f7f", "positive": "#2ca02c"}


def ensure_nltk_data():
    for pkg, path in [("stopwords", "corpora/stopwords"), ("vader_lexicon", "sentiment/vader_lexicon.zip"),
                      ("wordnet", "corpora/wordnet.zip"), ("punkt", "tokenizers/punkt")]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)


# --------------------------------------------------------------------------- #
# 1. Data
# --------------------------------------------------------------------------- #
def load_split(name: str) -> pd.DataFrame:
    with open(os.path.join(DATA_DIR, f"{name}_text.txt"), encoding="utf-8") as f:
        texts = [line.rstrip("\n") for line in f]
    with open(os.path.join(DATA_DIR, f"{name}_labels.txt")) as f:
        labels = [int(line) for line in f if line.strip()]
    df = pd.DataFrame({"text": texts, "label": labels})
    df["sentiment"] = df["label"].map(dict(enumerate(LABELS)))
    return df


# --------------------------------------------------------------------------- #
# 2. NLTK preprocessing
# --------------------------------------------------------------------------- #
tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
lemmatizer = WordNetLemmatizer()
STOP = set(stopwords.words("english")) - {"not", "no", "nor", "never", "but"}  # keep negations


def clean_tweet(text: str) -> str:
    text = re.sub(r"http\S+|www\.\S+", " ", text)      # links
    text = re.sub(r"@\w+", " ", text)                   # mentions
    text = text.replace("#", "")                        # keep hashtag words
    tokens = tokenizer.tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t.isalpha() and t not in STOP and len(t) > 1]
    return " ".join(tokens)


# --------------------------------------------------------------------------- #
# 3. Exploratory analysis
# --------------------------------------------------------------------------- #
def run_eda(train: pd.DataFrame):
    print("=== Dataset ===")
    print(train["sentiment"].value_counts())

    plt.figure(figsize=(5.5, 4))
    order = LABELS
    ax = sns.countplot(x="sentiment", data=train, order=order, hue="sentiment", palette=PALETTE, legend=False)
    for p in ax.patches:
        ax.annotate(int(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height()), ha="center", va="bottom")
    plt.title("Tweets per sentiment class (training set)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "class_distribution.png"), dpi=150)
    plt.close()

    train["n_words"] = train["text"].str.split().str.len()
    plt.figure(figsize=(7, 4))
    sns.boxplot(x="sentiment", y="n_words", data=train, order=order, hue="sentiment", palette=PALETTE, legend=False)
    plt.title("Tweet length (words) by sentiment")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "tweet_length.png"), dpi=150)
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, lab in zip(axes, order):
        words = Counter(" ".join(train.loc[train["sentiment"] == lab, "clean"]).split())
        top = pd.DataFrame(words.most_common(15), columns=["word", "count"])
        sns.barplot(data=top, y="word", x="count", color=PALETTE[lab], ax=ax)
        ax.set_title(f"Top words - {lab}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "top_words.png"), dpi=150)
    plt.close()


# --------------------------------------------------------------------------- #
# 4. Models
# --------------------------------------------------------------------------- #
def vader_predict(texts, sia, pos_thr=0.05, neg_thr=-0.05):
    scores = np.array([sia.polarity_scores(t)["compound"] for t in texts])
    pred = np.where(scores >= pos_thr, 2, np.where(scores <= neg_thr, 0, 1))
    return pred, scores


def make_features():
    return FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True, max_features=200000)),
    ])


def build_models():
    return {
        "TF-IDF + Logistic Regression": Pipeline([
            ("features", make_features()),
            ("clf", LogisticRegression(C=2.0, max_iter=2000, class_weight="balanced")),
        ]),
        "TF-IDF + Linear SVM": Pipeline([
            ("features", make_features()),
            ("clf", LinearSVC(C=0.3, class_weight="balanced")),
        ]),
    }


def main():
    ensure_nltk_data()
    train, test = load_split("train"), load_split("test")
    print(f"Train tweets: {len(train)}   Test tweets: {len(test)}")

    print("Cleaning tweets with NLTK (tokenize, remove stopwords, lemmatize)...")
    train["clean"] = train["text"].apply(clean_tweet)
    test["clean"] = test["text"].apply(clean_tweet)
    print("Example:")
    print("  raw  :", train.loc[2, "text"][:100])
    print("  clean:", train.loc[2, "clean"][:100])
    run_eda(train)

    results, reports, fitted = [], [], {}

    # --- VADER baseline ---------------------------------------------------- #
    sia = SentimentIntensityAnalyzer()
    v_pred, v_scores = vader_predict(test["text"], sia)
    results.append({"model": "VADER (rule-based)", "accuracy": accuracy_score(test["label"], v_pred),
                    "macro_f1": f1_score(test["label"], v_pred, average="macro")})
    reports.append("### VADER (rule-based)\n" + classification_report(test["label"], v_pred, target_names=LABELS))
    print("\n=== VADER (rule-based) ===")
    print(classification_report(test["label"], v_pred, target_names=LABELS))

    plt.figure(figsize=(7, 4))
    for lab_id, lab in enumerate(LABELS):
        sns.kdeplot(v_scores[test["label"].values == lab_id], label=f"true {lab}", color=PALETTE[lab], fill=True, alpha=0.25)
    plt.axvline(0.05, ls="--", c="k", lw=1)
    plt.axvline(-0.05, ls="--", c="k", lw=1)
    plt.xlabel("VADER compound score")
    plt.title("VADER score distribution by true sentiment\n(the classes overlap a lot - why a trained model helps)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "vader_scores.png"), dpi=150)
    plt.close()

    # --- trained models ---------------------------------------------------- #
    for name, model in build_models().items():
        print(f"\nTraining {name} ...")
        model.fit(train["clean"], train["label"])
        pred = model.predict(test["clean"])
        fitted[name] = model
        results.append({"model": name, "accuracy": accuracy_score(test["label"], pred),
                        "macro_f1": f1_score(test["label"], pred, average="macro")})
        reports.append(f"### {name}\n" + classification_report(test["label"], pred, target_names=LABELS))
        print(f"=== {name} ===")
        print(classification_report(test["label"], pred, target_names=LABELS))

    res = pd.DataFrame(results).set_index("model")
    print("\n=== Comparison on the 12 284-tweet test set ===")
    print(res.round(4))

    res.plot(kind="bar", figsize=(8, 4.5), rot=0, ylim=(0, 1), color=["#4C72B0", "#DD8452"])
    plt.title("Accuracy and macro-F1 on the test set")
    plt.ylabel("score")
    plt.legend(loc="upper left")
    for i, (a, f) in enumerate(zip(res["accuracy"], res["macro_f1"])):
        plt.text(i - 0.13, a + 0.01, f"{a:.2f}", ha="center", fontsize=9)
        plt.text(i + 0.13, f + 0.01, f"{f:.2f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "model_comparison.png"), dpi=150)
    plt.close()

    best_name = res.loc[list(fitted)]["macro_f1"].idxmax()
    best = fitted[best_name]
    cm = confusion_matrix(test["label"], best.predict(test["clean"]))
    ConfusionMatrixDisplay(cm, display_labels=LABELS).plot(cmap="Blues")
    plt.title(f"Confusion matrix - {best_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()

    joblib.dump(best, os.path.join(OUT_DIR, "sentiment_model.joblib"))
    with open(os.path.join(OUT_DIR, "metrics.txt"), "w") as f:
        f.write(res.round(4).to_string() + "\n\n" + "\n".join(reports))
    print(f"\nBest model: {best_name}  ->  saved to outputs/sentiment_model.joblib")

    samples = [
        "Absolutely loving the new update, everything runs so smooth!",
        "Flight delayed again. Three hours at the gate and no information. Worst airline ever.",
        "The meeting has been moved to Thursday at 3pm.",
        "Not bad, but I expected more for the price.",
    ]
    print("\n=== Predictions on new tweets ===")
    for s, p in zip(samples, best.predict([clean_tweet(s) for s in samples])):
        print(f"[{LABELS[p]:8s}] {s}")


if __name__ == "__main__":
    main()
