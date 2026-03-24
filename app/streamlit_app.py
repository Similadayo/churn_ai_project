import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
from PIL import Image
import plotly.express as px

# ==============================
# Page Configuration
# ==============================
st.set_page_config(
    page_title="🛒 Customer Churn Intelligence Suite",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================
# Load Model (Pipeline)
# ==============================
MODEL_PATH = os.path.join("models", "churn_model.pkl")

@st.cache_resource
def load_model():
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return None

model = load_model()

if not model:
    st.stop()

# ==============================
# Sidebar Navigation
# ==============================
st.sidebar.title("📂 Navigation")
page = st.sidebar.radio("Navigate to:", ["🏠 Predictions", "📊 Explainability", "📈 Trends & Segmentation"])

# ==============================
# PAGE 1 — Predictions
# ==============================
if page == "🏠 Predictions":
    st.title("🛒 Customer Churn Prediction Dashboard")
    st.markdown("Upload your customer data to **predict churn risk** and uncover actionable insights.")

    uploaded_file = st.file_uploader(
        "📂 Upload customer CSV (same columns as data/customers.csv, except 'churned')",
        type=["csv"]
    )

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.info(f"✅ {df.shape[0]} records loaded successfully!")

            expected_cols = [
                'customer_id', 'total_spent', 'avg_order_value', 'num_visits',
                'recency_days', 'complaint_flag', 'membership_level', 'region',
                'support_tickets', 'days_since_signup'
            ]
            missing_cols = [col for col in expected_cols if col not in df.columns]
            if missing_cols:
                st.error(f"Missing columns: {missing_cols}")
                st.stop()

            X = df.drop('customer_id', axis=1)

            # Predict
            with st.spinner("🔮 Running churn predictions..."):
                churn_proba = model.predict_proba(X)[:, 1]
                churn_pred = (churn_proba >= 0.35).astype(int)

            results = df.copy()
            results["Churn Probability"] = np.round(churn_proba, 3)
            results["Churn Prediction"] = np.where(churn_pred == 1, "Yes", "No")

            # Save for Trends & Segmentation tab
            st.session_state["results_df"] = results

            # Results Table
            st.subheader("📊 Prediction Results")
            st.dataframe(results[["customer_id", "Churn Prediction", "Churn Probability"]])

            churn_rate = (results["Churn Prediction"] == "Yes").mean() * 100
            avg_prob = results["Churn Probability"].mean() * 100

            col1, col2 = st.columns(2)
            col1.metric("Predicted Churn Rate", f"{churn_rate:.1f}%")
            col2.metric("Average Churn Probability", f"{avg_prob:.1f}%")

            st.subheader("🔍 Churn Probability Distribution")
            st.bar_chart(results["Churn Probability"])

            # Top 10 At-Risk Customers
            st.subheader("🚨 Top 10 At-Risk Customers")
            top_churners = results.sort_values("Churn Probability", ascending=False).head(10)

            def highlight_high_risk(row):
                color = "#ffcccc" if row["Churn Probability"] >= 0.6 else "#fff2cc"
                return [f"background-color: {color}" for _ in row]

            styled_top = top_churners[[
                "customer_id", "membership_level", "region", "total_spent",
                "num_visits", "recency_days", "Churn Probability"
            ]].style.apply(highlight_high_risk, axis=1)

            st.dataframe(styled_top, use_container_width=True)
            st.caption("🔴 Red = high risk (> 60%) | 🟡 Yellow = medium risk (35–60%)")

            # AI-style Insight Summary
            st.subheader("💡 Insight Summary")
            if churn_rate < 10:
                risk_level = "🟢 **Low churn risk overall.**"
            elif churn_rate < 30:
                risk_level = "🟡 **Moderate churn risk detected.**"
            else:
                risk_level = "🔴 **High churn risk! Immediate retention action required.**"

            common_level = results.loc[results["Churn Prediction"] == "Yes", "membership_level"].mode()
            common_region = results.loc[results["Churn Prediction"] == "Yes", "region"].mode()

            membership_trend = common_level.iloc[0] if not common_level.empty else "N/A"
            region_trend = common_region.iloc[0] if not common_region.empty else "N/A"

            st.markdown(f"""
            {risk_level}  
            • **Churn Rate:** {churn_rate:.1f}% of customers likely to churn.  
            • **Average Churn Probability:** {avg_prob:.1f}%  
            • **Top Risk Segment:** {membership_trend}-tier, {region_trend} region.  
            • **Recommended Action:** Launch retention offers, feedback follow-ups, and loyalty incentives.  
            """)

            csv = results.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Predictions as CSV",
                data=csv,
                file_name="churn_predictions.csv",
                mime="text/csv"
            )

            st.success("✅ Predictions complete! Explore trends or explainability next.")

        except Exception as e:
            st.error(f"❌ Error processing file: {e}")
    else:
        st.warning("Please upload a customer dataset (CSV) to proceed.")

# ==============================
# PAGE 2 — Explainability
# ==============================
elif page == "📊 Explainability":
    st.title("📊 Model Explainability & Feature Insights")
    st.markdown("Understand **why** customers churn using SHAP explainability analysis.")

    reports_dir = "reports"
    shap_summary_path = os.path.join(reports_dir, "shap_summary.png")
    shap_beeswarm_path = os.path.join(reports_dir, "shap_beeswarm.png")

    if not (os.path.exists(shap_summary_path) and os.path.exists(shap_beeswarm_path)):
        st.warning("⚠️ SHAP reports not found. Please run `python shap_explain.py` to generate them.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📈 SHAP Summary Plot")
            st.image(Image.open(shap_summary_path), caption="Feature influence summary", use_container_width=True)
        with col2:
            st.subheader("🐝 SHAP Beeswarm Plot")
            st.image(Image.open(shap_beeswarm_path), caption="Customer-level impact", use_container_width=True)

        st.subheader("💡 Key Takeaways")
        st.markdown("""
        - **Number of Visits**, **Recency Days**, and **Total Spent** are top churn predictors.  
        - **Silver-tier members** in specific regions show higher churn risk.  
        - **Complaints** and **low engagement** correlate strongly with churn.  
        - These insights can guide retention and loyalty strategies.
        """)
        st.success("✅ SHAP explainability results displayed successfully.")

# ==============================
# PAGE 3 — Trends & Segmentation
# ==============================
elif page == "📈 Trends & Segmentation":
    st.title("📈 Churn Trends & Customer Segmentation")
    st.markdown("Analyze churn distribution across demographics and behavioral segments.")

    if "results_df" not in st.session_state:
        st.warning("⚠️ Please run predictions first to view trends.")
        st.stop()

    results = st.session_state["results_df"].copy()

    # ==============================
    # Segmentation Visuals
    # ==============================
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🏷️ Churn by Membership Level")
        fig1 = px.bar(
            results.groupby("membership_level")["Churn Prediction"].apply(lambda x: (x == "Yes").mean() * 100).reset_index(),
            x="membership_level",
            y="Churn Prediction",
            color="membership_level",
            title="Churn Rate by Membership Tier (%)",
            text_auto=".1f"
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.subheader("🌍 Churn by Region")
        fig2 = px.bar(
            results.groupby("region")["Churn Prediction"].apply(lambda x: (x == "Yes").mean() * 100).reset_index(),
            x="region",
            y="Churn Prediction",
            color="region",
            title="Churn Rate by Region (%)",
            text_auto=".1f"
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ==============================
    # Behavioral Correlation
    # ==============================
    st.subheader("💵 Spending vs Churn Probability")
    fig3 = px.scatter(
        results,
        x="total_spent",
        y="Churn Probability",
        color="membership_level",
        title="Spending vs Churn Probability by Membership",
        hover_data=["region", "num_visits"]
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("""
    **Insight Summary:**  
    - Customers with lower spending and fewer visits show higher churn likelihood.  
    - Loyalty programs should target Basic-tier customers in high-churn regions.  
    - Increasing engagement frequency significantly reduces churn risk.
    """)

    st.success("✅ Segmentation analysis complete.")
