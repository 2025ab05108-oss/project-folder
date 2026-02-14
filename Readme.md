
# Bank Marketing Subscription Prediction – ML Assignment 2

## Course Information
- **Program:** M.Tech (AIML / DSE)  
- **Course:** Machine Learning  
- **Assignment:** Assignment – 2  
- **Marks:** 15  

---

## a. Problem Statement

The objective of this assignment is to design, evaluate, and deploy multiple machine
learning classification models to predict whether a customer will subscribe to a term
deposit using marketing campaign data.

The project demonstrates a complete **end-to-end machine learning workflow**:
- Dataset preprocessing and encoding  
- Implementation of multiple classification models  
- Evaluation using standard performance metrics  
- Deployment of an interactive Streamlit web application  

This is a **binary classification problem**, where:

- `1` → Customer subscribed  
- `0` → Customer did not subscribe  

---

## b. Dataset Description  **[1 Mark]**

- **Dataset Name:** Bank Marketing Dataset  
- **File Used:** `bank.csv`  
- **Problem Type:** Binary Classification  

### Dataset Characteristics
- **Number of Instances:** 41,188  
- **Number of Features:** 21  
- **Target Variable:** `y`  
  - `1` → Subscription  
  - `0` → No Subscription  

### Data Preprocessing & Cleansing
As implemented in `app.py`, the following preprocessing steps were applied **before
model training**:

- Label encoding for categorical variables  
- Train-test split (80:20 ratio)  
- Feature scaling using **StandardScaler** (for Logistic Regression and KNN)  

These steps ensure realistic model evaluation and prevent data leakage.

---

## c. Models Used & Evaluation Metrics  **[6 Marks]**

All models were trained and evaluated on the **same processed dataset**.

### Implemented Models
1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbors (KNN)  
4. Naive Bayes (Gaussian)  
5. Random Forest (Ensemble Model)  
6. XGBoost (Ensemble Model)  

### Evaluation Metrics
Each model was evaluated using:
- Accuracy  
- AUC Score  
- Precision  
- Recall  
- F1 Score  
- Matthews Correlation Coefficient (MCC)  

---

## 📊 Model Comparison Table

| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|----------|----------|-----|-----------|--------|----------|-----|
| Logistic Regression | 0.91 | 0.91 | 0.67 | 0.42 | 0.51 | 0.51 |
| Decision Tree | 0.89 | 0.89 | 0.51 | 0.51 | 0.51 | 0.51 |
| KNN | 0.90 | 0.90 | 0.59 | 0.40 | 0.47 | 0.47 |
| Naive Bayes | 0.85 | 0.85 | 0.40 | 0.61 | 0.48 | 0.48 |
| Random Forest | 0.91 | 0.91 | 0.65 | 0.51 | 0.57 | 0.57 |
| XGBoost | 0.92 | 0.92 | 0.65 | 0.55 | 0.60 | 0.60 |

---

## 🧠 Observations on Model Performance  **[3 Marks]**

| ML Model | Observation |
|----------|------------|
| Logistic Regression | Provides a strong baseline accuracy (91%) but shows low recall (0.42) for the minority class. |
| Decision Tree | Balanced performance but slightly lower accuracy (89%) and may overfit compared to ensemble models. |
| KNN | Moderate performance; lower recall indicates difficulty detecting positive class instances. |
| Naive Bayes | Lowest overall accuracy (85%) but highest recall (0.61), detecting more subscribers. |
| Random Forest | Strong ensemble model with improved F1-score and better variance control. |
| XGBoost | Best overall performance (92% accuracy, highest F1-score 0.60) with balanced precision and recall. |

---

## 🌐 Streamlit Application Features

The Streamlit application includes:
- CSV dataset upload option  
- GitHub dataset auto-download option  
- Model selection dropdown  
- Display of all required evaluation metrics  
- Confusion matrix visualization  
- Classification report display  

---

## 🗂️ Project Structure

project-folder/  
│-- app.py  
│-- requirements.txt  
│-- README.md  
│-- bank.csv  
│-- model/  
    ├── logistic_regression_model.py  
    ├── decision_tree_model.py  
    ├── knn_model.py  
    ├── naive_bayes_model.py  
    ├── random_forest_model.py  
    └── xgboost_model.py  

---

## ⚙️ How to Run the Application

### Install Dependencies
pip install -r requirements.txt

### Run Streamlit App
streamlit run app.py

---

## 🧪 Execution Environment

- The assignment was executed using Python 3.10  
- Deployment performed on Streamlit Community Cloud  

---

## 📜 Academic Integrity Declaration

This assignment has been independently implemented in accordance with the
Academic Integrity Guidelines. AI tools were used only for conceptual understanding
and learning support.

---

## Submission Details

Platform Used: Streamlit Community Cloud  

Python Version Used: 3.10  

Deployment Type: Free Tier  

GitHub Repository: https://github.com/2025ab05108-oss/project-folder  

Live Application URL: https://project-folder-iucedghan984k7bqdrufkx.streamlit.app

---

## ✅ Final Submission Checklist

- GitHub repository link works  
- Streamlit app link opens correctly  
- All six models implemented  
- Evaluation metrics displayed  
- README.md included in submitted PDF  
- Comparison table included  
- Observations table included  
