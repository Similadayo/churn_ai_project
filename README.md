# 🛒 Customer Churn Prediction using Machine Learning

This project implements a **Predictive Analytics and Machine Learning system** for forecasting customer churn in e-commerce businesses.
It is designed as a **research and development prototype**, demonstrating how data-driven models can identify customers at risk of leaving a platform
and support retention strategies.

---

## 🚀 Project Overview

The system uses **Python, scikit-learn, Streamlit, and SHAP** to create an end-to-end pipeline for:

- Data preprocessing and balancing (using SMOTE)
- Training and evaluating multiple ML models
- Model interpretability with SHAP explainability
- Interactive prediction and analytics via a Streamlit dashboard

---

## 🎯 Objectives

- Build a predictive model to identify likely churners.
- Preprocess and balance datasets effectively.
- Compare Logistic Regression and Random Forest models.
- Evaluate accuracy, recall, F1-score, and ROC-AUC.
- Visualize churn insights and top risk customers.
- Explain model behavior with SHAP plots.
- Deploy results in an interactive Streamlit web app.

---

## 🧠 System Workflow

1. **Data Generation** – Synthetic customer dataset created for experiments.
2. **Preprocessing** – Cleaning, scaling, encoding, and balancing with SMOTE.
3. **Model Training** – Compare Logistic Regression & Random Forest.
4. **Evaluation** – Confusion matrices, ROC curves, and metrics comparison.
5. **Explainability** – SHAP plots for model transparency.
6. **Visualization** – Streamlit dashboard with 3 tabs (Predictions, Explainability, Trends).

---

## 🗂️ Project Structure

```
churn_ai_project/
│
├── data/
│   └── customers.csv
├── models/
│   └── churn_model.pkl
├── reports/
│   ├── shap_summary.png
│   └── shap_beeswarm.png
├── app/
│   └── streamlit_app.py
├── generate_data.py
├── train_model.py
├── shap_explain.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation Guide

### 1️⃣ Create Virtual Environment

```bash
python -m venv venv
# Activate (Windows)
venv\Scripts\activate
# Activate (Mac/Linux)
source venv/bin/activate
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Generate Data (Optional)

```bash
python generate_data.py
```

### 4️⃣ Train Model

```bash
python train_model.py
```

### 5️⃣ Run Explainability (Optional but recommended)

```bash
python shap_explain.py
```

### 6️⃣ Launch Streamlit Dashboard

```bash
streamlit run app/streamlit_app.py
```

---

## 📊 Dashboard Overview

### 🏠 **Predictions Tab**

- Upload a CSV file (`customers.csv` without `churned` column).
- View churn predictions and risk probabilities.
- See top 10 high-risk customers.
- Download churn results as CSV.

### 📈 **Explainability Tab**

- Visualize SHAP summary and beeswarm plots.
- Understand how each feature affects churn probability.

### 🌍 **Trends & Segmentation Tab**

- Explore churn rate by membership level, region, and spending behavior.
- Gain insights into customer engagement patterns.

---

## 📘 Technologies Used

| Category                       | Tools                                  |
| ------------------------------ | -------------------------------------- |
| **Programming Language** | Python                                 |
| **Machine Learning**     | scikit-learn, imbalanced-learn         |
| **Visualization**        | Streamlit, Plotly, Seaborn, Matplotlib |
| **Explainability**       | SHAP                                   |
| **Model Persistence**    | Joblib                                 |

---

## 📈 Model Performance Summary

| Metric   | Logistic Regression | Random Forest |
| -------- | ------------------- | ------------- |
| Accuracy | ~85%                | ~90%          |
| Recall   | Moderate            | High          |
| ROC-AUC  | 0.88                | 0.93          |

> *Random Forest* was selected as the final model due to higher recall and AUC performance.

---

## 🧭 Key Insights

- **Recency**, **Number of Visits**, and **Total Spending** are top predictors of churn.
- **Silver-tier** members and customers with fewer interactions are more likely to churn.
- Balancing and proper feature scaling significantly improved model stability.

---

## 🧩 For Academic Use

This system is intended for **final-year research projects** and **academic demonstrations**.
It showcases the **complete machine learning lifecycle** — from data generation to visualization and interpretation.
It aligns with academic report Chapters 1–5 (Problem Statement → Methodology → Results → Discussion → Conclusion).

---

## 📬 Author & Credits

Developed by: **[Your Name / Team Name]**
Advisor / Supervisor: *[Insert Supervisor Name]*
Institution: *[Insert School or Department]*
Year: *2025*

---

### 🏁 License

This project is open for educational and non-commercial use.
