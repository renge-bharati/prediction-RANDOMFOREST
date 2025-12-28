import streamlit as st
import pickle
import numpy as np

st.set_page_config(page_title="Prediction App", layout="centered")
st.title("🔮 Prediction App (PKL Model)")

# ---------- Load Model ----------
@st.cache_resource
def load_model():
    with open("df2.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()
st.success("✅ Model loaded successfully")

# ---------- Detect Features ----------
if hasattr(model, "n_features_in_"):
    n_features = model.n_features_in_
else:
    st.error("❌ Cannot detect number of input features from model")
    st.stop()

# ---------- Input Section ----------
st.subheader("📥 Enter Input Features")

inputs = []
for i in range(n_features):
    val = st.number_input(f"Feature {i+1}", value=0.0)
    inputs.append(val)

# ---------- Prediction ----------
if st.button("🔍 Predict"):
    data = np.array(inputs).reshape(1, -1)
    prediction = model.predict(data)
    st.success(f"📌 Prediction: {prediction[0]}")
s
