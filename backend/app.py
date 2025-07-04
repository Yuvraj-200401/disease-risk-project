from flask import Flask, request, jsonify
from flask_cors import CORS
from utils.predictor import predict_disease

app = Flask(__name__)
CORS(app)  # Allow frontend to connect

@app.route("/")
def home():
    return jsonify({"message": "Disease Risk Predictor API is running."})

@app.route("/predict", methods=["POST"])
def predict():
    print("📩 Received a prediction request!") 
    data = request.get_json()

    disease = data.get("disease")
    features = data.get("features")

    if not disease or not features:
        return jsonify({"error": "Missing data"}), 400

    result = predict_disease(disease, features)
    return jsonify(result)
if __name__ == '__main__':
    from os import environ
    app.run(host='0.0.0.0', port=int(environ.get("PORT", 5000)))