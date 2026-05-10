import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="FINSCORE Credit Scoring", layout="wide")

# -----------------------------
# LOAD MODEL
# -----------------------------
model = joblib.load("models/model_lgb.pkl")

st.title("FINSCORE: BNPL-Aware Credit Scoring System")

st.write("""
This tool estimates a consumer's **FINSCORE** using behavioral financial indicators such as  
BNPL usage, credit utilization, cash advances, and spending patterns.
""")

# -----------------------------
# INPUT SECTION
# -----------------------------

st.subheader("Enter Financial Details")
st.write("Use the inputs below to capture income, credit usage, BNPL behavior, and your current loan position.")

col1, col2, col3 = st.columns(3)

with col1:

    age = st.slider("Age",21,65,35)

    employment_type = st.selectbox(
        "Employment Type",
        ["Salaried","Self Employed"]
    )

    income = st.number_input(
        "Monthly Income (₹)",
        min_value=10000,
        max_value=500000,
        value=50000
    )

    total_spend = st.number_input(
        "Total Monthly Spending (₹)",
        min_value=1000,
        max_value=300000,
        value=20000
    )

    bnpl_total = st.number_input(
        "BNPL Spending (₹)",
        min_value=0,
        max_value=200000,
        value=5000
    )

with col2:

    credit_cards = st.slider(
        "Number of Credit Cards",
        1,5,2
    )

    city_tier = st.selectbox(
        "City Tier",
        ["Tier 1","Tier 2","Tier 3"]
    )

    credit_limit_total = st.number_input(
        "Total Credit Limit (₹)",
        min_value=10000,
        max_value=1000000,
        value=200000
    )

    credit_cash_total = st.number_input(
        "Credit Card Cash Withdrawal (₹)",
        min_value=0,
        max_value=50000,
        value=0
    )

    credit_cash_transactions = st.slider(
        "Cash Advance Transactions",
        0,15,0
    )

with col3:

    external_support_total = st.number_input(
        "Money Borrowed from Family/Friends (₹)",
        min_value=0,
        max_value=50000,
        value=0
    )

    external_support_transactions = st.slider(
        "External Support Transactions",
        0,10,0
    )

    current_active_loan = st.number_input(
        "Current Active Loan Balance (₹)",
        min_value=0,
        max_value=2000000,
        value=0
    )

    min_balance = st.number_input(
        "Minimum Account Balance (₹)",
        min_value=-50000,
        max_value=200000,
        value=10000
    )

    bnpl_transactions = st.slider(
    "BNPL Transactions",
    0,20,2
    )

# -----------------------------
# FEATURE ENGINEERING
# -----------------------------

estimated_txn_count = max(1, bnpl_transactions + credit_cash_transactions + external_support_transactions + 8)
avg_spend = total_spend / estimated_txn_count
spend_volatility = avg_spend * 0.8
max_spend = max(avg_spend * 1.8, total_spend * 0.25)

bnpl_usage_ratio = bnpl_total / total_spend if total_spend > 0 else 0

cash_advance_ratio = credit_cash_total / total_spend if total_spend > 0 else 0
external_support_ratio = external_support_total / income if income > 0 else 0
loan_income_ratio = current_active_loan / income if income > 0 else 0

bnpl_income_ratio = bnpl_total / income if income > 0 else 0
credit_utilisation = max_spend / credit_limit_total if credit_limit_total > 0 else 0

# Additional features required by model
spend_income_ratio = total_spend / income if income > 0 else 0
overspend_ratio = max(0, (total_spend - income) / income) if income > 0 else 0
debt_to_income = current_active_loan / income if income > 0 else 0
buffer_ratio = min_balance / income if income > 0 else 0
# Fixed: use actual credit usage (spending) instead of loan balance for revolving utilisation
revolving_utilisation = (total_spend / credit_limit_total) if credit_limit_total > 0 else 0
payment_ratio = max(0, 1 - loan_income_ratio) if income > 0 else 0
payment_delay = int(min_balance < 0)
bnpl_installment_ratio = bnpl_total / income if income > 0 else 0
credit_history_years = (age - 21) / 10  # Approximate
hard_inquiry_count = int(credit_cards > 3)

# Behavior segments
segment_cash_dependent = int(cash_advance_ratio > 0.25)
segment_external_support = int(external_support_ratio > 1)

employment_type_self_employed = 1 if employment_type == "Self Employed" else 0
city_tier_tier2 = 1 if city_tier == "Tier 2" else 0
city_tier_tier3 = 1 if city_tier == "Tier 3" else 0

# -----------------------------
# MODEL INPUT
# -----------------------------

input_df = pd.DataFrame([{

    "age": age,
    "monthly_income": income,
    "credit_cards": credit_cards,
    "credit_limit_total": credit_limit_total,

    "total_spend": total_spend,
    "avg_spend": avg_spend,
    "spend_volatility": spend_volatility,
    "max_spend": max_spend,

    "bnpl_total": bnpl_total,
    "bnpl_transactions": bnpl_transactions,

    "min_balance": min_balance,
    "bnpl_usage_ratio": bnpl_usage_ratio,

    "credit_cash_total": credit_cash_total,
    "credit_cash_transactions": credit_cash_transactions,

    "external_support_total": external_support_total,
    "external_support_transactions": external_support_transactions,

    "cash_advance_ratio": cash_advance_ratio,
    "external_support_ratio": external_support_ratio,

    "employment_type_self_employed": employment_type_self_employed,
    "city_tier_tier2": city_tier_tier2,
    "city_tier_tier3": city_tier_tier3,

    "segment_cash_dependent": segment_cash_dependent,
    "segment_external_support": segment_external_support,

    # Additional features required by model
    "current_active_loan": current_active_loan,
    "loan_income_ratio": loan_income_ratio,
    "spend_income_ratio": spend_income_ratio,
    "overspend_ratio": overspend_ratio,
    "payment_ratio": payment_ratio,
    "payment_delay": payment_delay,
    "bnpl_income_ratio": bnpl_income_ratio,
    "credit_utilisation": credit_utilisation,
    "debt_to_income": debt_to_income,
    "buffer_ratio": buffer_ratio,
    "revolving_utilisation": revolving_utilisation,
    "credit_history_years": credit_history_years,
    "hard_inquiry_count": hard_inquiry_count,
    "bnpl_installment_ratio": bnpl_installment_ratio

}])

# -----------------------------
# PREDICTION
# -----------------------------

if st.button("Calculate FINSCORE"):

    feature_order = model.calibrated_classifiers_[0].estimator.feature_names_in_
    input_df = input_df[feature_order]

    prob_default = model.predict_proba(input_df)[0][1]

    # -----------------------------
    # BEHAVIORAL FINSCORE CALCULATION
    # -----------------------------
    # More nuanced scoring that considers behavioral patterns beyond just default probability
    
    # Base score from probability (300-900 range)
    base_score = 900 - (prob_default * 600)
    
    # Behavioral adjustments
    behavioral_adjustment = 0
    
    # BNPL dependency penalty
    if bnpl_income_ratio > 0.4:
        behavioral_adjustment -= 50
    elif bnpl_income_ratio > 0.25:
        behavioral_adjustment -= 25
    
    # Credit utilization penalty
    if credit_utilisation > 0.8:
        behavioral_adjustment -= 40
    elif credit_utilisation > 0.6:
        behavioral_adjustment -= 20
    
    # Payment history bonus/penalty
    if payment_delay == 1:
        behavioral_adjustment -= 30
    elif min_balance > income * 0.5:
        behavioral_adjustment += 15  # Strong buffer
    
    # Overspending penalty
    if overspend_ratio > 0.2:
        behavioral_adjustment -= 35
    elif overspend_ratio > 0.1:
        behavioral_adjustment -= 15
    
    # Cash advance dependency penalty
    if segment_cash_dependent == 1:
        behavioral_adjustment -= 25
    
    # External borrowing penalty
    if segment_external_support == 1:
        behavioral_adjustment -= 20
    
    # Debt burden penalty
    if loan_income_ratio > 2.0:
        behavioral_adjustment -= 40
    elif loan_income_ratio > 1.0:
        behavioral_adjustment -= 20
    
    # Age-based stability bonus (older = more stable)
    if age > 45:
        behavioral_adjustment += 10
    elif age < 28:
        behavioral_adjustment -= 10
    
    # Employment stability bonus
    if employment_type == "Salaried":
        behavioral_adjustment += 5
    
    # Apply behavioral adjustments
    finscore = int(np.clip(base_score + behavioral_adjustment, 300, 900))

    if finscore >= 800:
        category = "Excellent"
    elif finscore >= 740:
        category = "Very Good"
    elif finscore >= 670:
        category = "Good"
    elif finscore >= 580:
        category = "Risky"
    else:
        category = "High Risk"

    # -----------------------------
    # BEHAVIOR PROFILE
    # -----------------------------

    if segment_cash_dependent:
        behavior = "Cash Advance Dependent"

    elif segment_external_support:
        behavior = "External Borrowing"

    elif bnpl_income_ratio > 0.35:
        behavior = "BNPL Dependent"

    elif credit_utilisation > 0.75:
        behavior = "Credit Card Revolver"

    elif min_balance < 0:
        behavior = "Liquidity Stressed"

    else:
        behavior = "Financially Stable"

    # -----------------------------
    # EXPLANATION
    # -----------------------------

    reasons = []

    if bnpl_income_ratio > 0.35:
        reasons.append("High BNPL dependence relative to income")

    if credit_utilisation > 0.75:
        reasons.append("High credit utilization")

    if cash_advance_ratio > 0.25:
        reasons.append("Frequent credit card cash withdrawals")

    if external_support_ratio > 1:
        reasons.append("Dependence on external financial support")

    if min_balance < 0:
        reasons.append("Negative account balance detected")

    if total_spend > income:
        reasons.append("Spending exceeds monthly income")
    
    if len(reasons) == 0:
        reasons.append("Financial behavior appears stable with no significant risk patterns detected")

    # if len(reasons) == 0:
    #     if prob_default > 0.7:
    #         reasons.append("Model detected statistical risk patterns in historical data")
    #     else:
    #         reasons.append("Financial behavior appears stable")

    # -----------------------------
    # OUTPUT
    # -----------------------------

    st.divider()

    st.subheader("FINSCORE Result")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("FINSCORE", finscore)
    c2.metric("Risk Category", category)
    c3.metric("Default Probability", round(prob_default,3))
    c4.metric("Current Loan Going", f"₹{int(current_active_loan):,}")

    st.subheader("Loan Summary")
    l1, l2 = st.columns(2)
    l1.metric("Loan / Income", f"{loan_income_ratio:.2%}")
    l2.metric("Current Loan Balance", f"₹{int(current_active_loan):,}")

    st.subheader("Behavior Profile")
    st.write(behavior)

    st.subheader("Explanation")

    for r in reasons:
        st.write("-", r)

    st.info("""
FINSCORE ranges from **300 to 900**

800–900 → Excellent credit health  
740–799 → Very good  
670–739 → Good  
580–669 → Risky  
Below 580 → High default risk
""")