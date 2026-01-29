import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests

# ---------------- CONFIG ----------------
API_BASE = "https://soil-quality-backend.vercel.app/api"

st.set_page_config(
    page_title="Soil Quality Prediction System",
    page_icon="🌱",
    layout="wide"
)

# ---------------- UI ----------------
st.title("🌱 Soil Quality Prediction System")

st.markdown(
    "Predict **Soil Quality (Low / Medium / High)** using a Machine Learning model "
    "served via **FastAPI (Vercel)** and consumed by **Streamlit**."
)

# ---------------- SIDEBAR ----------------
st.sidebar.header("🔧 Input Soil Parameters")

nitrogen = st.sidebar.slider("Nitrogen (ppm)", 0.0, 200.0, 50.0)
phosphorus = st.sidebar.slider("Phosphorus (ppm)", 0.0, 200.0, 40.0)
potassium = st.sidebar.slider("Potassium (ppm)", 0.0, 200.0, 30.0)
ph = st.sidebar.slider("Soil pH", 0.0, 14.0, 6.8)
moisture = st.sidebar.slider("Moisture (%)", 0.0, 100.0, 25.0)

# ---------------- PREDICTION ----------------
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
        res = requests.post(f"{API_BASE}/predict", json=payload, timeout=10)
        res.raise_for_status()
        result = res.json()

        st.success(f"✅ Predicted Soil Quality: **{result['soil_quality']}**")
        st.info(f"Confidence: **{result['confidence']*100:.2f}%**")

        prob_df = pd.DataFrame(
            result["probabilities"].items(),
            columns=["Soil Quality", "Probability"]
        ).set_index("Soil Quality")

        st.subheader("📈 Prediction Probabilities")
        st.bar_chart(prob_df)

    except Exception as e:
        st.error("❌ Backend prediction failed")
        st.code(str(e))

# ---------------- METRICS ----------------
st.markdown("---")
st.subheader("📊 Model Performance Metrics")

try:
    metrics_res = requests.get(f"{API_BASE}/metrics", timeout=10)
    metrics_res.raise_for_status()
    metrics = metrics_res.json()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", metrics["accuracy"])
    col2.metric("Precision", metrics["precision"])
    col3.metric("Recall", metrics["recall"])
    col4.metric("F1 Score", metrics["f1_score"])

    st.subheader("🧮 Confusion Matrix")

    cm = np.array(metrics["confusion_matrix"])
    labels = metrics["labels"]

    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

except Exception:
    st.warning("⚠️ Model metrics are temporarily unavailable (backend waking up).")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("🚀 Powered by FastAPI + Streamlit + Machine Learning")
