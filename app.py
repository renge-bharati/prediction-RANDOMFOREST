import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Page config
st.set_page_config(page_title="Prediction App", layout="centered")

st.title("🔮 Machine Learning Prediction App")

# Load model
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

st.success("Model loaded successfully ✅")

# Try to get feature names
feature_names = None

if hasattr(model, "feature_names_in_"):
    feature_names = model.feature_names_in_
else:
    st.warning("Feature names not found in model. Using manual input.")
    feature_names = [f"Feature {i+1}" for i in range(model.n_features_in_)]

st.subheader("📥 Enter Input Values")

input_data = []

for feature in feature_names:
    value = st.number_input(f"{feature}", value=0.0)
    input_data.append(value)

input_array = np.array(input_data).reshape(1, -1)

# Prediction
if st.button("🚀 Predict"):
    try:
        prediction = model.predict(input_array)
        st.success(f"✅ Prediction Result: **{prediction[0]}**")
    except Exception as e:
        st.error("Prediction failed ❌")
        st.write(e)
