import streamlit as st
import requests

st.set_page_config(page_title="Disease Risk Predictor", page_icon="🏥")

st.title("🏥 Disease Risk Predictor + Health Tracker")
st.write("Enter your health details to predict your disease risk.")

# Backend API URL
API_URL = "http://localhost:5000/predict"

disease = st.selectbox("🔍 Select a disease to predict:", ["heart", "diabetes", "stroke"])

features = []

# Feature inputs per disease
if disease == "heart":
    st.subheader("🫀 Heart Disease Inputs")
    features.append(st.slider("Age", 20, 80, 50))
    features.append(st.selectbox("Sex", [0, 1]))
    features.append(st.slider("Resting BP", 80, 200, 120))
    features.append(st.slider("Cholesterol", 100, 400, 220))
    features.append(st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1]))
    features.append(st.slider("Max Heart Rate Achieved", 60, 202, 150))
    features.append(st.selectbox("Exercise-induced Angina", [0, 1]))
    features += [0, 1, 0, 0]  # Dummy values for remaining fields (if needed)

elif disease == "diabetes":
    st.subheader("🩸 Diabetes Inputs")
    features.append(st.slider("Pregnancies", 0, 15, 2))
    features.append(st.slider("Glucose", 0, 200, 120))
    features.append(st.slider("BloodPressure", 0, 150, 70))
    features.append(st.slider("SkinThickness", 0, 100, 20))
    features.append(st.slider("Insulin", 0, 900, 80))
    features.append(st.slider("BMI", 10.0, 60.0, 25.0))
    features.append(st.slider("Diabetes Pedigree Function", 0.0, 3.0, 0.5))
    features.append(st.slider("Age", 10, 100, 33))

elif disease == "stroke":
    st.subheader("🧠 Stroke Prediction Inputs")
    features.append(st.slider("Age", 1, 100, 45))
    features.append(st.selectbox("Hypertension (1 = yes)", [0, 1]))
    features.append(st.selectbox("Heart Disease", [0, 1]))
    features.append(st.selectbox("Ever Married", [0, 1]))
    features.append(st.selectbox("Work Type (0 = Private, 1 = Govt, etc.)", [0, 1, 2, 3, 4]))
    features.append(st.selectbox("Residence Type (0 = Urban, 1 = Rural)", [0, 1]))
    features.append(st.slider("Avg Glucose Level", 50, 300, 100))
    features.append(st.slider("BMI", 10.0, 60.0, 25.0))
    features.append(st.selectbox("Smoking Status (0 = never, 1 = former, 2 = smokes)", [0, 1, 2]))

if st.button("🔮 Predict"):
    with st.spinner("Predicting..."):
        try:
            response = requests.post(API_URL, json={
                "disease": disease,
                "features": features
            })

            if response.status_code == 200:
                result = response.json()
                st.success(f"🧾 Prediction: {'❗ Risk Detected' if result['prediction'] else '✅ No Risk'}")
                st.info(f"🧠 Confidence: {result['confidence'] * 100:.2f}%")

                # Show health tip
                if result['prediction'] == 1:
                    st.warning("💡 Health Tip:")
                    if disease == "heart":
                        st.write("- Reduce sodium & cholesterol\n- Do regular cardio")
                    elif disease == "diabetes":
                        st.write("- Reduce sugar & carb intake\n- Exercise daily")
                    elif disease == "stroke":
                        st.write("- Avoid smoking\n- Manage blood pressure & stress")
            else:
                st.error("Something went wrong. Please try again.")

        except Exception as e:
            st.error(f"Error: {e}")
