import joblib
import numpy as np

# Load all 3 models
models = {
    "heart": joblib.load("model/heart_model.pkl"),
    "diabetes": joblib.load("model/diabetes_model.pkl"),
    "stroke": joblib.load("model/stroke_model.pkl"),
}

def predict_disease(disease, input_data):
    if disease not in models:
        return {"error": "Unknown disease type."}

    model = models[disease]
    input_array = np.array(input_data).reshape(1, -1)

    prediction = model.predict(input_array)[0]
    probability = model.predict_proba(input_array)[0].max()

    return {
        "prediction": int(prediction),
        "confidence": round(float(probability), 2)
    }
