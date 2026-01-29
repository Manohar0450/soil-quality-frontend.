import streamlit as st
import requests
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

API_BASE = "https://soil-quality-backend.vercel.app/api"

st.set_page_config(
    page_title="Soil Quality Prediction",
    page_icon="🌱",
    layout="wide"
)

st.title("🌱 Soil Quality Prediction System")

# ---------------- Sidebar ----------------
st.sidebar.header("🔧 Input Soil Parameters")

nitrogen = st.sidebar.slider("Nitrogen (ppm)", 0.0, 200.0, 50.0)
phosphorus = st.sidebar.slider("Phosphorus (ppm)", 0.0, 200.0, 40.0)
potassium = st.sidebar.slider("Potassium (ppm)", 0.0, 200.0, 30.0)
ph = st.sidebar.slider("Soil pH", 0.0, 14.0, 6.8)
moisture = st.sidebar.slider("Moisture (%)", 0.0, 100.0, 25.0)

# ---------------- Prediction ----------------
if st.button("🔍 Predict Soil Quality"):
    payload = {
        "nitrogen": nitrogen,
        "phosphorus": phosphorus,
        "potassium": potassium,
        "ph": ph,
        "moisture": moisture
    }

    res = requests.post(f"{API_BASE}/predict", json=payload)
    data = res.json()

    st.success(f"**Soil Quality:** {data['soil_quality']}")
    st.info(f"**Confidence:** {data['confidence'] * 100:.2f}%")

    st.subheader("📈 Prediction Probabilities")
    prob_df = pd.DataFrame.from_dict(
        data["probabilities"], orient="index", columns=["Probability"]
    )
    st.bar_chart(prob_df)

# ---------------- Model Metrics ----------------
st.markdown("---")
st.subheader("📊 Model Performance Metrics")

metrics = requests.get(f"{API_BASE}/metrics").json()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", metrics["accuracy"])
col2.metric("Precision", metrics["precision"])
col3.metric("Recall", metrics["recall"])
col4.metric("F1 Score", metrics["f1_score"])

# ---------------- Confusion Matrix ----------------
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

st.caption("🚀 Powered by FastAPI + Streamlit + Machine Learning")
