import streamlit as st
import pickle
import numpy as np

st.set_page_config(page_title="Prediction App", layout="centered")
st.title("🔮 ML Prediction App (Using PKL Model)")

# Load model
@st.cache_resource
def load_model():
    with open("df2.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()
st.success("✅ Model loaded successfully")

# Get number of features from model
if hasattr(model, "n_features_in_"):
    n_features = model.n_features_in_
else:
    st.error("❌ Cannot detect number of features from model")
    st.stop()

st.subheader("📥 Enter Input Features")

inputs = []
for i in range(n_features):
    value = st.number_input(f"Feature {i+1}", value=0.0)
    inputs.append(value)

# Predict button
if st.button("🔍 Predict"):
    input_array = np.array(inputs).reshape(1, -1)
    prediction = model.predict(input_array)

    st.subheader("📌 Prediction Result")
    st.success(f"Prediction: {prediction[0]}")
