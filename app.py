import streamlit as st
import requests
import pandas as pd

# ---------------- Page Configuration ----------------
st.set_page_config(
    page_title="Soil Quality Prediction System",
    page_icon="🌱",
    layout="wide"
)

# ---------------- Backend URL ----------------
API_URL = "https://soil-quality-backend.vercel.app/api/predict"

# ---------------- Title ----------------
st.title("🌱 Soil Quality Prediction System")
st.markdown(
    "Predict **Soil Quality (Low / Medium / High)** using a Machine Learning model "
    "served via **FastAPI (Vercel)** and consumed by **Streamlit**."
)

# ---------------- Sidebar Inputs ----------------
st.sidebar.header("🔧 Input Soil Parameters")

nitrogen = st.sidebar.slider("Nitrogen (ppm)", 0.0, 200.0, 50.0)
phosphorus = st.sidebar.slider("Phosphorus (ppm)", 0.0, 200.0, 40.0)
potassium = st.sidebar.slider("Potassium (ppm)", 0.0, 200.0, 30.0)
ph = st.sidebar.slider("Soil pH", 0.0, 14.0, 6.8)
moisture = st.sidebar.slider("Moisture (%)", 0.0, 100.0, 25.0)

# ---------------- Predict Button ----------------
st.subheader("🔍 Predict Soil Quality")

if st.button("Predict Soil Quality"):
    payload = {
        "nitrogen": nitrogen,
        "phosphorus": phosphorus,
        "potassium": potassium,
        "ph": ph,
        "moisture": moisture
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=10)

        if response.status_code == 200:
            result = response.json()

            st.success(f"✅ Predicted Soil Quality: **{result['soil_quality']}**")
            st.info(f"Confidence: **{result['confidence'] * 100:.2f}%**")

            # Probability chart
            prob_df = pd.DataFrame.from_dict(
                result["probabilities"], orient="index", columns=["Probability"]
            )
            st.subheader("📈 Prediction Probabilities")
            st.bar_chart(prob_df)

        else:
            st.error("❌ Backend returned an error")
            st.json(response.json())

    except Exception as e:
        st.error("❌ Backend API error")
        st.code(str(e))

# ---------------- Footer ----------------
st.markdown("---")
st.caption("🚀 Powered by FastAPI + Streamlit + Machine Learning")
