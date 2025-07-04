from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

# Load models
heart_model = joblib.load("model/heart_model.pkl")
diabetes_model = joblib.load("model/diabetes_model.pkl")
stroke_model = joblib.load("model/stroke_model.pkl")

@app.route("/")
def index():
    return jsonify({"message": "Disease Risk Predictor API is running."})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    disease = data.get("disease")
    features = data.get("features")

    if not disease or features is None:
        return jsonify({"error": "Missing input"}), 400

    # Pad dummy values if needed for heart model
    if disease == "heart":
        expected_features = 13
        if len(features) < expected_features:
            features += [0] * (expected_features - len(features))  # Add 0s to match

    X = np.array(features).reshape(1, -1)

    if disease == "heart":
        model = heart_model
    elif disease == "diabetes":
        model = diabetes_model
    elif disease == "stroke":
        model = stroke_model
    else:
        return jsonify({"error": "Invalid disease type"}), 400

    try:
        prediction = model.predict(X)[0]
        confidence = model.predict_proba(X).max()

        return jsonify({
            "prediction": int(prediction),
            "confidence": round(float(confidence), 4)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Required for Render deployment
if __name__ == '__main__':
    from os import environ
    app.run(host='0.0.0.0', port=int(environ.get("PORT", 5000)))
