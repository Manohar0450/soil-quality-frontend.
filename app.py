import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

# ---------------- Page Configuration ----------------
st.set_page_config(
    page_title="Soil Quality Prediction System",
    page_icon="🌱",
    layout="wide"
)

# ---------------- Title & Description ----------------
st.title("🌱 Soil Quality Prediction System")
st.markdown("""
This intelligent system predicts **soil quality (Low / Medium / High)** using machine learning.
It helps farmers and agritech professionals make **data-driven decisions** for better crop planning,
efficient fertilizer usage, and sustainable agriculture.
""")

# ---------------- Sidebar Inputs ----------------
st.sidebar.header("🔧 Input Soil Parameters")

nitrogen = st.sidebar.slider("Nitrogen (ppm)", 0.0, 200.0, 50.0, help="Nitrogen content in soil")
phosphorus = st.sidebar.slider("Phosphorus (ppm)", 0.0, 200.0, 40.0, help="Phosphorus content in soil")
potassium = st.sidebar.slider("Potassium (ppm)", 0.0, 200.0, 30.0, help="Potassium content in soil")
ph = st.sidebar.slider("Soil pH", 0.0, 14.0, 6.8, help="Soil acidity/alkalinity")
moisture = st.sidebar.slider("Moisture (%)", 0.0, 100.0, 25.0, help="Soil moisture percentage")

if st.sidebar.button("🔄 Reset Inputs"):
    st.rerun()

# ---------------- Generate Dataset ----------------
np.random.seed(42)
rows_per_class = 200

data = {"Nitrogen": [], "Phosphorus": [], "Potassium": [], "pH": [], "Moisture": [], "Soil_Quality": []}

def generate_data(label, n):
    for _ in range(n):
        if label == "Low":
            data["Nitrogen"].append(np.random.uniform(10, 25))
            data["Phosphorus"].append(np.random.uniform(10, 25))
            data["Potassium"].append(np.random.uniform(20, 35))
            data["pH"].append(np.random.uniform(4.5, 5.9))
            data["Moisture"].append(np.random.uniform(10, 15))
        elif label == "Medium":
            data["Nitrogen"].append(np.random.uniform(25, 35))
            data["Phosphorus"].append(np.random.uniform(20, 35))
            data["Potassium"].append(np.random.uniform(35, 45))
            data["pH"].append(np.random.uniform(6.0, 6.4))
            data["Moisture"].append(np.random.uniform(15, 20))
        else:  # High
            data["Nitrogen"].append(np.random.uniform(35, 50))
            data["Phosphorus"].append(np.random.uniform(30, 45))
            data["Potassium"].append(np.random.uniform(45, 60))
            data["pH"].append(np.random.uniform(6.5, 7.5))
            data["Moisture"].append(np.random.uniform(20, 30))

        # Add small noise (realistic accuracy)
        if np.random.rand() < 0.05:
            data["Soil_Quality"].append(np.random.choice(["Low","Medium","High"]))
        else:
            data["Soil_Quality"].append(label)

for lbl in ["Low", "Medium", "High"]:
    generate_data(lbl, rows_per_class)

df = pd.DataFrame(data)
df.to_csv("soil_data.csv", index=False)

# ---------------- Encode Target ----------------
le = LabelEncoder()
df["Soil_Quality"] = le.fit_transform(df["Soil_Quality"])

X = df[["Nitrogen","Phosphorus","Potassium","pH","Moisture"]]
y = df["Soil_Quality"]

# ---------------- Train-Test Split ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- Train Model ----------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model & encoder
with open("soil_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

# ---------------- Prediction ----------------
st.subheader("🔍 Soil Quality Prediction")

if st.button("Predict Soil Quality"):

    # ✅ FIX: Use DataFrame with feature names
    input_data = pd.DataFrame(
        [[nitrogen, phosphorus, potassium, ph, moisture]],
        columns=["Nitrogen", "Phosphorus", "Potassium", "pH", "Moisture"]
    )

    with open("soil_model.pkl","rb") as f:
        loaded_model = pickle.load(f)
    with open("label_encoder.pkl","rb") as f:
        loaded_le = pickle.load(f)

    prediction = loaded_model.predict(input_data)
    confidence = loaded_model.predict_proba(input_data).max()
    soil_quality = loaded_le.inverse_transform(prediction)[0]

    st.success(f"✅ Predicted Soil Quality: **{soil_quality}**")
    st.info(f"Prediction Confidence: **{confidence*100:.2f}%**")

    # ---------------- Model Evaluation ----------------
    y_pred = loaded_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    st.subheader("📊 Model Performance")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{accuracy*100:.2f}%")
    col2.metric("Precision", f"{precision:.2f}")
    col3.metric("Recall", f"{recall:.2f}")
    col4.metric("F1 Score", f"{f1:.2f}")

    # ---------------- Confusion Matrix ----------------
    # ---------------- Confusion Matrix ----------------
    st.subheader("🧮 Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(3, 2.5))  # 🔽 further reduced
    sns.heatmap(
        cm,
        annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=loaded_le.classes_,
    yticklabels=loaded_le.classes_,
    cbar=False,          # remove color bar
    annot_kws={"size": 8},  # smaller numbers
    ax=ax
    )
    ax.set_xlabel("Predicted", fontsize=9)
    ax.set_ylabel("Actual", fontsize=9)
    ax.tick_params(labelsize=8)
    st.pyplot(fig)


    # ---------------- Classification Report (UPDATED) ----------------
    st.subheader("📋 Classification Report")
    report_text = classification_report(
        y_test,
        y_pred,
        target_names=loaded_le.classes_,
        zero_division=0
    )
    st.text(report_text)

    # ---------------- Feature Importance ----------------
    st.subheader("🌿 Feature Importance")

    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": loaded_model.feature_importances_
    }).sort_values(by="Importance", ascending=True)

    fig2, ax2 = plt.subplots(figsize=(4, 2.5))  # ✅ smaller
    ax2.barh(importance["Feature"], importance["Importance"])
    ax2.set_xlabel("Importance")
    ax2.set_ylabel("Feature")
    ax2.tick_params(labelsize=9)
    st.pyplot(fig2)

    # ---------------- Prediction Probability ----------------
    st.subheader("📈 Prediction Probabilities")

    prob_df = pd.DataFrame(
        loaded_model.predict_proba(input_data)[0],
        index=loaded_le.classes_,
        columns=["Probability"]
    )
    st.bar_chart(prob_df)