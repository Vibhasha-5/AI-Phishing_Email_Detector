import joblib
import argparse
import sys
import os

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "../models/phishing_model.joblib")

def load_model():
    return joblib.load(MODEL_PATH)

def main():
    parser = argparse.ArgumentParser(
        description="Phishing Email Detection (CLI)"
    )
    parser.add_argument(
        "-t", "--text",
        help="Email text to classify",
        required=False
    )
    args = parser.parse_args()

    if args.text:
        email_text = args.text
    else:
        print("Paste email content, then press Ctrl+D:")
        email_text = sys.stdin.read()

    model = load_model()
    prediction = model.predict([email_text])[0]
    probability = model.predict_proba([email_text])[0]

    label = "PHISHING" if prediction == 1 else "LEGITIMATE"

    print("\nPrediction:", label)
    print("Confidence:", probability)

if __name__ == "__main__":
    main()
