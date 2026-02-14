import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

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

# Import XGBoost from model folder
from model.xgboost_model import train_xgboost


# ===============================
# Page Config
# ===============================

st.set_page_config(page_title="Bank Marketing ML App", layout="wide")

st.title("📊 Bank Marketing Classification Application")
st.markdown("### Hari Prasad K C - 2025AB05108 - BITS ML Assignment 2")


# ===============================
# Load Dataset
# ===============================

@st.cache_data
def load_data():
    return pd.read_csv("bank.csv", sep=";")

df = load_data()

# ===============================
# Sidebar Upload Option
# ===============================

st.sidebar.header("Upload Test Dataset (CSV)")
uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, sep=";")
    st.sidebar.success("Dataset uploaded successfully!")

# ===============================
# Dataset Overview
# ===============================

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

# ===============================
# Preprocessing
# ===============================

df_processed = df.copy()

for col in df_processed.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df_processed[col] = le.fit_transform(df_processed[col])

if "y" not in df_processed.columns:
    st.error("Target column 'y' not found.")
    st.stop()

X = df_processed.drop("y", axis=1)
y = df_processed["y"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# Model Selection
# ===============================

st.sidebar.header("Select Classification Model")

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

# ===============================
# Model Evaluation
# ===============================

st.header("🤖 Model Evaluation")

if st.button("Run Model"):

    # Scaling only for LR and KNN
    if model_option in ["Logistic Regression", "KNN"]:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test

    # Select Model
    if model_option == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train_scaled, y_train)

    elif model_option == "Decision Tree":
        model = DecisionTreeClassifier(random_state=42)
        model.fit(X_train_scaled, y_train)

    elif model_option == "KNN":
        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(X_train_scaled, y_train)

    elif model_option == "Naive Bayes":
        model = GaussianNB()
        model.fit(X_train_scaled, y_train)

    elif model_option == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)

    elif model_option == "XGBoost (Ensemble)":
        model, X_test_scaled = train_xgboost(X_train_scaled, y_train, X_test_scaled)

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

    # ===============================
    # Display Metrics
    # ===============================

    st.subheader("📈 Evaluation Metrics")

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", round(accuracy, 4))
    col2.metric("AUC Score", round(auc, 4))
    col3.metric("Precision", round(precision, 4))

    col4, col5, col6 = st.columns(3)
    col4.metric("Recall", round(recall, 4))
    col5.metric("F1 Score", round(f1, 4))
    col6.metric("MCC", round(mcc, 4))

    # Confusion Matrix
    st.subheader("🔲 Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig)

    # Classification Report
    st.subheader("📋 Classification Report")
    st.text(classification_report(y_test, y_pred))
