import pandas as pd
import joblib
import shap
import os
import numpy as np
import matplotlib.pyplot as plt

# ==============================
# Paths
# ==============================
MODEL_PATH = os.path.join('models', 'churn_model.pkl')
DATA_PATH = os.path.join('data', 'customers.csv')
REPORTS_DIR = 'reports'
os.makedirs(REPORTS_DIR, exist_ok=True)

# ==============================
# Load model and dataset
# ==============================
print("Loading model and dataset...")
model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)

if 'customer_id' in df.columns:
    df = df.drop('customer_id', axis=1)

y = df['churned']
X = df.drop('churned', axis=1)

# ==============================
# Extract pipeline components
# ==============================
try:
    preprocessor = model.named_steps['preprocessor']
    classifier = model.named_steps['classifier']
except KeyError as e:
    raise KeyError("Pipeline step not found. Ensure your pipeline uses steps named 'preprocessor' and 'classifier'.") from e

X_processed = preprocessor.transform(X)
feature_names = preprocessor.get_feature_names_out()
X_sample = X_processed[:200]

# ==============================
# Compute SHAP values
# ==============================
print("Computing SHAP values...")
if hasattr(classifier, "estimators_"):  # RandomForest or tree-based
    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(X_sample)
    shap_array = shap_values[1] if isinstance(shap_values, list) else shap_values
elif hasattr(classifier, "coef_"):  # Logistic Regression
    explainer = shap.LinearExplainer(classifier, X_sample)
    shap_array = explainer.shap_values(X_sample)
else:
    explainer = shap.Explainer(classifier, X_sample)
    shap_array = explainer(X_sample).values

# ==============================
# Generate and save SHAP plots
# ==============================
print("Generating SHAP plots...")

# --- SHAP Summary Plot ---
plt.figure()
shap.summary_plot(shap_array, features=X_sample, feature_names=feature_names, show=False)
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, 'shap_summary.png'), dpi=200)
plt.close()

# --- SHAP Beeswarm Plot ---
plt.figure()
shap.summary_plot(shap_array, features=X_sample, feature_names=feature_names, plot_type="dot", show=False)
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, 'shap_beeswarm.png'), dpi=200)
plt.close()

# ==============================
# Feature Importance Summary
# ==============================
print("\n✅ SHAP analysis complete. Reports saved to 'reports/'.")

mean_abs_shap = np.abs(shap_array).mean(axis=0)
top_features_idx = np.argsort(mean_abs_shap)[-5:][::-1]
top_features = [(feature_names[i], mean_abs_shap[i]) for i in top_features_idx]

print("\n💡 Top 5 most influential features:")
for feature, val in top_features:
    val_scalar = float(np.ravel(val)[0])  # convert to scalar safely
    print(f" - {feature}: {val_scalar:.4f}")
