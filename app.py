import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests

# ================= CONFIG =================
API_BASE = "https://soil-quality-backend.vercel.app/api"

st.set_page_config(
    page_title="Soil Quality Prediction System",
    page_icon="🌱",
    layout="wide"
)

# ================= TITLE =================
st.title("🌱 Soil Quality Prediction System")
st.markdown(
    "Predict **Soil Quality (Low / Medium / High)** using a Machine Learning model "
    "served via **FastAPI (Vercel)** and consumed by **Streamlit**."
)

# ================= SIDEBAR INPUTS =================
st.sidebar.header("🔧 Input Soil Parameters")

nitrogen = st.sidebar.slider("Nitrogen (ppm)", 0.0, 200.0, 50.0)
phosphorus = st.sidebar.slider("Phosphorus (ppm)", 0.0, 200.0, 40.0)
potassium = st.sidebar.slider("Potassium (ppm)", 0.0, 200.0, 30.0)
ph = st.sidebar.slider("Soil pH", 0.0, 14.0, 6.8)
moisture = st.sidebar.slider("Moisture (%)", 0.0, 100.0, 25.0)

if st.sidebar.button("🔄 Reset"):
    st.rerun()

# ================= PREDICT =================
st.subheader("🔍 Predict Soil Quality")

payload = {
    "nitrogen": nitrogen,
    "phosphorus": phosphorus,
    "potassium": potassium,
    "ph": ph,
    "moisture": moisture
}

if st.button("🔮 Predict Soil Quality"):
    try:
        response = requests.post(f"{API_BASE}/predict", json=payload)

        if response.status_code == 200:
            result = response.json()

            st.success(f"✅ Predicted Soil Quality: **{result['soil_quality']}**")
            st.info(f"Confidence: **{result['confidence']*100:.2f}%**")

            # -------- Probability Chart --------
            st.subheader("📈 Prediction Probabilities")
            prob_df = pd.DataFrame.from_dict(
                result["probabilities"],
                orient="index",
                columns=["Probability"]
            )
            st.bar_chart(prob_df)

        else:
            st.error("❌ Backend returned an error")
            st.json(response.json())

    except Exception as e:
        st.error("❌ Backend API error")
        st.exception(e)

# ================= METRICS =================
st.divider()
st.subheader("📊 Model Performance Metrics")

try:
    metrics_res = requests.get(f"{API_BASE}/metrics")

    if metrics_res.status_code == 200:
        metrics = metrics_res.json()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", metrics["accuracy"])
        col2.metric("Precision", metrics["precision"])
        col3.metric("Recall", metrics["recall"])
        col4.metric("F1 Score", metrics["f1_score"])

        # -------- Confusion Matrix --------
        st.subheader("🧮 Confusion Matrix")

        cm = pd.DataFrame(
            metrics["confusion_matrix"],
            index=metrics["labels"],
            columns=metrics["labels"]
        )

        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

    else:
        st.error("❌ Failed to fetch metrics from backend")

except Exception as e:
    st.error("❌ Metrics API error")
    st.exception(e)

# ================= FOOTER =================
st.markdown("---")
st.caption("🚀 Powered by FastAPI + Streamlit + Machine Learning")
