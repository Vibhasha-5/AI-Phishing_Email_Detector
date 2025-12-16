import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import accuracy_score, classification_report

from feature_utils import url_feature_transformer

BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "../data/phishing_email.csv")
MODEL_PATH = os.path.join(BASE_DIR, "../models/phishing_model.joblib")

def main():
    df = pd.read_csv(DATA_PATH)

    X = df["text_combined"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    text_features = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=50000,
        stop_words="english"
    )

    url_features = FunctionTransformer(
    url_feature_transformer,
    validate=False
    
    )

    combined_features = FeatureUnion([
        ("tfidf", text_features),
        ("url", url_features)
    ])

    model = Pipeline([
        ("features", combined_features),
        ("clf", LogisticRegression(
            max_iter=3000,
            class_weight="balanced"
        ))
    ])

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, preds))
    print("\nClassification Report:\n")
    print(classification_report(y_test, preds))

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print("\nModel saved at:", MODEL_PATH)

if __name__ == "__main__":
    main()
