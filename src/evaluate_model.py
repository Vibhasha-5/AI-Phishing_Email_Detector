import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay
)

BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "../data/phishing_email.csv")
MODEL_PATH = os.path.join(BASE_DIR, "../models/phishing_model.joblib")

def main():
    df = pd.read_csv(DATA_PATH)
    X = df["text_combined"]
    y = df["label"]

    model = joblib.load(MODEL_PATH)

    probs = model.predict_proba(X)[:, 1]
    preds = model.predict(X)

    roc_auc = roc_auc_score(y, probs)
    print("ROC-AUC Score:", roc_auc)

    fpr, tpr, _ = roc_curve(y, probs)
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.3f})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.show()

    cm = confusion_matrix(y, preds)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.show()

if __name__ == "__main__":
    main()
