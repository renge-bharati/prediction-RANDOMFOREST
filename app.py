import streamlit as st
import pickle
import numpy as np
import os

st.set_page_config(page_title="Prediction App", layout="centered")
st.title("🔮 Prediction App")

# Absolute path fix
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("❌ model.pkl not found!")
        st.stop()
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

model = load_model()
st.success("✅ Model loaded successfully")

# Get feature count safely
if hasattr(model, "feature_names_in_"):
    features = model.feature_names_in_
else:
    features = [f"Feature {i+1}" for i in range(model.n_features_in_)]

st.subheader("📥 Input Features")

inputs = []
for feature in features:
    val = st.number_input(feature, value=0.0)
    inputs.append(val)

if st.button("🚀 Predict"):
    data = np.array(inputs).reshape(1, -1)
    prediction = model.predict(data)
    st.success(f"🎯 Prediction: **{prediction[0]}**")
