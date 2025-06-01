import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

# 1. Load models & data artifacts
@st.cache(allow_output_mutation=True)
def load_models():
    import joblib
    xgb_model = joblib.load("models/xgb_credit.pkl")
    rf_fraud   = joblib.load("models/rf_fraud.pkl")
    return xgb_model, rf_fraud

xgb_model, rf_fraud = load_models()

st.title("AI-Powered Financial Risk Platform")

# ── Credit Risk ─────────────────────────────────────────────────────────────────
st.header("Credit Risk Assessment")
# Sidebar inputs for loan features…
loan_amt = st.sidebar.number_input("Loan Amount", value=10000)
# … other inputs …

if st.sidebar.button("Compute PD"):
    X_input = pd.DataFrame({...}, index=[0])
    pd_prob = xgb_model.predict_proba(X_input)[:,1][0]
    st.metric("Default Probability", f"{pd_prob:.2%}")
    # SHAP explanation
    explainer = shap.TreeExplainer(xgb_model)
    sv = explainer.shap_values(X_input)[1]
    st.pyplot(shap.force_plot(
        explainer.expected_value[1], sv, X_input, matplotlib=True
    ))

# ── Portfolio Risk ───────────────────────────────────────────────────────────────
st.header("Portfolio Optimization")
# Load and plot efficient frontier
# …
st.pyplot()  # frontier
st.pyplot()  # cumulative backtest

# ── Fraud Detection ──────────────────────────────────────────────────────────────
st.header("Fraud Detection Alerts")
# Simulate a few transactions
df_alerts = pd.DataFrame(np.random.randn(5,10), columns=[f"feat_{i}" for i in range(10)])
probs     = rf_fraud.predict_proba(df_alerts)[:,1]
flags     = (probs >= 0.40)
df_alerts["Fraud?"] = flags
df_alerts["Score"] = probs
st.table(df_alerts)

st.success("Dashboard loaded successfully!")
