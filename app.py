import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import requests
from io import StringIO

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# =====================================================
# CONFIG
# =====================================================

st.set_page_config(page_title="Bank Marketing ML App", layout="wide")

st.title("📊 Bank Marketing Classification Application")
st.markdown("### Hari Prasad K C - 2025AB05108 - BITS ML Assignment 2")

# =====================================================
# GITHUB RAW URL
# =====================================================

GITHUB_RAW_URL = "https://raw.githubusercontent.com/2025ab05108-oss/project-folder/main/bank.csv"

# =====================================================
# SIDEBAR - DATASET SOURCE
# =====================================================

st.sidebar.header("📂 Dataset Source")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV Dataset",
    type=["csv"]
)

# ---- REAL DOWNLOAD BUTTON ----
st.sidebar.markdown("### Or Download Dataset from GitHub")

try:
    response = requests.get(GITHUB_RAW_URL)
    response.raise_for_status()
    csv_data = response.content

    st.sidebar.download_button(
        label="Download Dataset",
        data=csv_data,
        file_name="bank.csv",
        mime="text/csv"
    )

except:
    st.sidebar.error("Failed to fetch dataset from GitHub.")

# =====================================================
# LOAD DATASET
# =====================================================

@st.cache_data
def load_github_data():
    response = requests.get(GITHUB_RAW_URL)
    response.raise_for_status()
    return pd.read_csv(StringIO(response.text), sep=";")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, sep=";")
    st.sidebar.success("Using uploaded dataset.")
else:
    df = load_github_data()
    st.sidebar.info("Using dataset from GitHub repository.")

# =====================================================
# DATASET OVERVIEW
# =====================================================

st.header("📌 Dataset Overview")

col1, col2, col3 = st.columns(3)
col1.metric("Total Rows", df.shape[0])
col2.metric("Total Columns", df.shape[1])
col3.metric("Missing Values", df.isnull().sum().sum())

st.subheader("📋 Feature List")

feature_df = pd.DataFrame({
    "Feature Index": range(len(df.columns)),
    "Feature Name": df.columns
})

st.dataframe(feature_df, use_container_width=True)

st.subheader("🔎 Sample Records (First 5 Rows)")
st.dataframe(df.head(), use_container_width=True)

# =====================================================
# PREPROCESSING
# =====================================================

df_processed = df.copy()

for col in df_processed.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df_processed[col] = le.fit_transform(df_processed[col])

if "y" not in df_processed.columns:
    st.error("❌ Target column 'y' not found.")
    st.stop()

X = df_processed.drop("y", axis=1)
y = df_processed["y"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =====================================================
# MODEL SELECTION
# =====================================================

st.sidebar.header("🤖 Select Classification Model")

model_option = st.sidebar.selectbox(
    "Choose Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost (Ensemble)"
    ]
)

# =====================================================
# MODEL EVALUATION
# =====================================================

st.header("🤖 Model Evaluation")

if st.button("Run Model"):

    # Scaling where required
    if model_option in ["Logistic Regression", "KNN"]:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test

    # Model selection
    if model_option == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)

    elif model_option == "Decision Tree":
        model = DecisionTreeClassifier(random_state=42)

    elif model_option == "KNN":
        model = KNeighborsClassifier(n_neighbors=5)

    elif model_option == "Naive Bayes":
        model = GaussianNB()

    elif model_option == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)

    elif model_option == "XGBoost (Ensemble)":
        model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")

    # Train model
    model.fit(X_train_scaled, y_train)

    # Predictions
    y_pred = model.predict(X_test_scaled)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
    else:
        auc = 0

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    # =====================================================
    # DISPLAY METRICS
    # =====================================================

    st.subheader("📈 Evaluation Metrics")

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", round(accuracy, 4))
    col2.metric("AUC Score", round(auc, 4))
    col3.metric("Precision", round(precision, 4))

    col4, col5, col6 = st.columns(3)
    col4.metric("Recall", round(recall, 4))
    col5.metric("F1 Score", round(f1, 4))
    col6.metric("MCC", round(mcc, 4))

    # =====================================================
    # CONFUSION MATRIX
    # =====================================================

    st.subheader("🔲 Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig)

    # =====================================================
    # CLASSIFICATION REPORT
    # =====================================================

    st.subheader("📋 Classification Report")
    st.text(classification_report(y_test, y_pred))
