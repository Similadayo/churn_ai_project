import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, roc_curve
)
from imblearn.over_sampling import SMOTE
from joblib import dump
import seaborn as sns
import matplotlib.pyplot as plt

# ==============================
# Load dataset
# ==============================
DATA_PATH = os.path.join('data', 'customers.csv')
df = pd.read_csv(DATA_PATH)

# Drop unnecessary columns
if 'customer_id' in df.columns:
    df = df.drop('customer_id', axis=1)

# Features and target
y = df['churned']
X = df.drop('churned', axis=1)

# Identify categorical and numerical columns
cat_cols = ['membership_level', 'region']
num_cols = [col for col in X.columns if col not in cat_cols]

# ==============================
# Preprocessor setup
# ==============================
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_cols)
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==============================
# Preprocess and balance data
# ==============================
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("Before SMOTE:", np.bincount(y_train))
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train_processed, y_train)
print("After SMOTE:", np.bincount(y_train_bal))

# ==============================
# Define models
# ==============================
lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
rf = RandomForestClassifier(
    n_estimators=150, random_state=42, class_weight='balanced', max_depth=8
)

models = {
    'Logistic Regression': lr,
    'Random Forest': rf
}

results = {}
roc_curves = {}

# ==============================
# Train, evaluate, and threshold tuning
# ==============================
for name, model in models.items():
    print(f"\n=== Training {name} ===")
    model.fit(X_train_bal, y_train_bal)
    y_prob = model.predict_proba(X_test_processed)[:, 1]

    # Lower threshold → improve recall
    threshold = 0.35
    y_pred = (y_prob >= threshold).astype(int)

    # Evaluate
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    results[name] = {
        'model': model,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'roc_auc': roc_auc,
        'threshold': threshold
    }

    print(f"Accuracy: {acc:.3f} | Precision: {prec:.3f} | Recall: {rec:.3f} | F1: {f1:.3f} | AUC: {roc_auc:.3f}")
    print(classification_report(y_test, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'{name} (Threshold={threshold})')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_curves[name] = (fpr, tpr)

# ==============================
# ROC Curve comparison
# ==============================
plt.figure(figsize=(6, 5))
for name, (fpr, tpr) in roc_curves.items():
    plt.plot(fpr, tpr, label=f"{name} (AUC={results[name]['roc_auc']:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend()
plt.tight_layout()
plt.show()

# ==============================
# Save best model as full pipeline
# ==============================
best_model_name = max(results, key=lambda x: results[x]['roc_auc'])
best_estimator = results[best_model_name]['model']

# Build full pipeline for deployment
final_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', best_estimator)
])

# Fit pipeline on all available data (better for deployment)
final_pipeline.fit(X, y)

os.makedirs('models', exist_ok=True)
dump(final_pipeline, os.path.join('models', 'churn_model.pkl'))

print(f"\n✅ Best model: {best_model_name} (AUC={results[best_model_name]['roc_auc']:.3f})")
print("Full pipeline (preprocessor + model) saved to models/churn_model.pkl")
