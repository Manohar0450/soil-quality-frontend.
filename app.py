import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- Config ----------------
API_BASE = "https://soil-quality-backend.vercel.app/api"

st.set_page_config(
    page_title="Soil Quality Prediction System",
    page_icon="🌱",
    layout="wide"
)

# ---------------- Title ----------------
st.title("🌱 Soil Quality Prediction System")
st.caption("FastAPI (Vercel) + Streamlit + Machine Learning")

# ---------------- Sidebar Inputs ----------------
st.sidebar.header("🔧 Input Soil Parameters")

nitrogen = st.sidebar.slider("Nitrogen (ppm)", 0.0, 200.0, 50.0)
phosphorus = st.sidebar.slider("Phosphorus (ppm)", 0.0, 200.0, 40.0)
potassium = st.sidebar.slider("Potassium (ppm)", 0.0, 200.0, 30.0)
ph = st.sidebar.slider("Soil pH", 0.0, 14.0, 6.8)
moisture = st.sidebar.slider("Moisture (%)", 0.0, 100.0, 25.0)

# ---------------- Prediction ----------------
st.subheader("🔍 Predict Soil Quality")

if st.button("Predict Soil Quality"):
    payload = {
        "Nitrogen": nitrogen,
        "Phosphorus": phosphorus,
        "Potassium": potassium,
        "pH": ph,
        "Moisture": moisture
    }

    try:
        res = requests.post(f"{API_BASE}/predict", json=payload).json()

        st.success(f"🌱 Soil Quality: **{res['soil_quality']}**")
        st.info(f"Confidence: **{res['confidence'] * 100:.2f}%**")

        prob_df = pd.DataFrame.from_dict(
            res["probabilities"],
            orient="index",
            columns=["Probability"]
        )
        st.subheader("📈 Prediction Probabilities")
        st.bar_chart(prob_df)

    except Exception as e:
        st.error("Backend error")
        st.write(e)

# ---------------- Model Metrics ----------------
st.markdown("---")
st.subheader("📊 Model Performance Metrics")

try:
    metrics = requests.get(f"{API_BASE}/metrics").json()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", metrics["accuracy"])
    col2.metric("Precision", metrics["precision"])
    col3.metric("Recall", metrics["recall"])
    col4.metric("F1 Score", metrics["f1_score"])

    # Confusion Matrix
    st.subheader("🧮 Confusion Matrix")

    cm = metrics["confusion_matrix"]
    labels = metrics["labels"]

    fig, ax = plt.subplots()
    im = ax.imshow(cm)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i][j], ha="center", va="center")

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

except:
    st.warning("Metrics not available yet")
