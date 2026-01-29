import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt

# ================= CONFIG =================
API_URL = "https://soil-quality-backend.vercel.app/api/predict"

st.set_page_config(
    page_title="Soil Quality Prediction System",
    page_icon="🌱",
    layout="wide"
)

# ================= TITLE =================
st.title("🌱 Soil Quality Prediction System")
st.markdown("""
Predict **Soil Quality (Low / Medium / High)** using a Machine Learning model  
deployed on **FastAPI (Vercel)** and consumed via **Streamlit**.
""")

# ================= SIDEBAR INPUTS =================
st.sidebar.header("🔧 Input Soil Parameters")

nitrogen = st.sidebar.slider("Nitrogen (ppm)", 0.0, 200.0, 50.0)
phosphorus = st.sidebar.slider("Phosphorus (ppm)", 0.0, 200.0, 40.0)
potassium = st.sidebar.slider("Potassium (ppm)", 0.0, 200.0, 30.0)
ph = st.sidebar.slider("Soil pH", 0.0, 14.0, 6.8)
moisture = st.sidebar.slider("Moisture (%)", 0.0, 100.0, 25.0)

if st.sidebar.button("🔄 Reset"):
    st.rerun()

# ================= PREDICTION =================
st.subheader("🔍 Predict Soil Quality")

if st.button("Predict Soil Quality"):
    payload = {
        "Nitrogen": nitrogen,
        "Phosphorus": phosphorus,
        "Potassium": potassium,
        "pH": ph,
        "Moisture": moisture
    }

    with st.spinner("Contacting ML model..."):
        try:
            response = requests.post(API_URL, json=payload, timeout=15)
            response.raise_for_status()
            result = response.json()

            soil_quality = result["prediction"]
            confidence = result["confidence"]
            probabilities = result["probabilities"]

            st.success(f"✅ Predicted Soil Quality: **{soil_quality}**")
            st.info(f"Confidence: **{confidence:.2f}%**")

            # ================= PROBABILITY CHART =================
            st.subheader("📈 Prediction Probabilities")

            prob_df = pd.DataFrame.from_dict(
                probabilities, orient="index", columns=["Probability"]
            )

            st.bar_chart(prob_df)

        except Exception as e:
            st.error("❌ Backend API error")
            st.code(str(e))

# ================= FOOTER =================
st.markdown("---")
st.caption("🚀 Powered by FastAPI + Streamlit + Machine Learning")
