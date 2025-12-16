from flask import Flask, request, jsonify
import joblib
import os

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "../models/phishing_model.joblib")

API_KEY = "phishing-api-key-2025"  # stored as env var in production

app = Flask(__name__)
model = joblib.load(MODEL_PATH)

@app.route("/", methods=["GET"])
def home():
    return {
        "status": "Phishing Detection API is running",
        "secured": True,
        "endpoint": "/predict",
        "method": "POST"
    }

def authenticate(request):
    client_key = request.headers.get("x-api-key")
    return client_key == API_KEY

@app.route("/predict", methods=["POST"])
def predict():
    if not authenticate(request):
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json()

    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field"}), 400

    text = data["text"]

    prediction = model.predict([text])[0]
    probability = model.predict_proba([text])[0].tolist()

    return jsonify({
        "prediction": "PHISHING" if prediction == 1 else "LEGITIMATE",
        "probability": probability
    })

if __name__ == "__main__":
    app.run(debug=True)
