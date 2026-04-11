# app.py
import streamlit as st
import boto3
import json
import numpy as np

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "Insurance Pricing Engine",
    page_icon  = "🚗",
    layout     = "wide"
)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🚗 Insurance Frequency Pricing Engine")
st.markdown("*Powered by XGBoost — deployed on AWS SageMaker*")
st.divider()

# ── Sidebar — model info ──────────────────────────────────────────────────────
with st.sidebar:
    st.header("Model Info")
    st.metric("Model",       "XGBoost Frequency")
    st.metric("MAE",         "0.1925")
    st.metric("RMSE",        "0.5213")
    st.metric("Train rows",  "461,048")
    st.metric("Features",    "11")
    st.divider()
    st.caption("Model Registry: InsuranceFrequencyModels v1")
    st.caption("Endpoint: insurance-frequency-endpoint")
    st.caption("Region: us-east-2")

# ── Input form ────────────────────────────────────────────────────────────────
st.subheader("Policy Details")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Driver Profile**")
    driv_age    = st.slider("Driver Age",        min_value=18,  max_value=100, value=35)
    bonus_malus = st.slider("Bonus Malus",       min_value=50,  max_value=230, value=60,
                            help="50 = best driver, 230 = worst")
    area        = st.selectbox("Area",           ["A", "B", "C", "D", "E", "F"],
                               index=2,
                               help="A = rural, F = dense urban")
    density     = st.number_input("Population Density", min_value=1, max_value=27000,
                                  value=1500)

with col2:
    st.markdown("**Vehicle Profile**")
    veh_power   = st.slider("Vehicle Power",     min_value=4,   max_value=15,  value=6)
    veh_age     = st.slider("Vehicle Age",       min_value=0,   max_value=100, value=5)
    veh_gas     = st.selectbox("Fuel Type",      ["Regular", "Diesel"])

st.divider()

# ── Score button ──────────────────────────────────────────────────────────────
if st.button("Calculate Expected Frequency", type="primary", use_container_width=True):

    # ── Derive engineered features ────────────────────────────────────────────
    log_density = round(np.log1p(density), 4)
    high_power  = 1 if veh_power >= 9 else 0

    if driv_age <= 25:
        driv_age_group = "18-25"
    elif driv_age <= 35:
        driv_age_group = "26-35"
    elif driv_age <= 50:
        driv_age_group = "36-50"
    elif driv_age <= 65:
        driv_age_group = "51-65"
    else:
        driv_age_group = "65+"

    if veh_age <= 1:
        veh_age_group = "New"
    elif veh_age <= 5:
        veh_age_group = "1-5"
    elif veh_age <= 10:
        veh_age_group = "6-10"
    elif veh_age <= 20:
        veh_age_group = "11-20"
    else:
        veh_age_group = "Old"

    # ── Build policy payload ──────────────────────────────────────────────────
    policy = {
        "DrivAge"       : driv_age,
        "VehAge"        : veh_age,
        "VehPower"      : veh_power,
        "BonusMalus"    : bonus_malus,
        "Area"          : area,
        "VehGas"        : veh_gas,
        "Density"       : density,
        "DrivAge_Group" : driv_age_group,
        "VehAge_Group"  : veh_age_group,
        "High_Power"    : high_power,
        "Log_Density"   : log_density,
    }

    # ── Call endpoint ─────────────────────────────────────────────────────────
    try:
        runtime  = boto3.client('sagemaker-runtime', region_name='us-east-2')
        response = runtime.invoke_endpoint(
            EndpointName = "insurance-frequency-endpoint",
            ContentType  = "application/json",
            Body         = json.dumps(policy)
        )
        result    = json.loads(response['Body'].read())
        frequency = result['predictions'][0]

        # ── Display results ───────────────────────────────────────────────────
        st.divider()
        st.subheader("Prediction Results")

        col3, col4, col5 = st.columns(3)

        with col3:
            st.metric(
                label = "Expected Claims / Year",
                value = f"{frequency:.4f}",
            )

        with col4:
            st.metric(
                label = "Expected Claims / Year %",
                value = f"{frequency * 100:.2f}%",
            )

        with col5:
            if frequency < 0.10:
                risk_tier  = "Low Risk"
                risk_color = "green"
            elif frequency < 0.20:
                risk_tier  = "Medium Risk"
                risk_color = "orange"
            elif frequency < 0.40:
                risk_tier  = "High Risk"
                risk_color = "red"
            else:
                risk_tier  = "Very High Risk"
                risk_color = "darkred"

            st.metric(label="Risk Tier", value=risk_tier)

        # ── Policy summary ────────────────────────────────────────────────────
        with st.expander("View full policy details sent to model"):
            st.json(policy)

        # ── Actuarial context ─────────────────────────────────────────────────
        st.info(
            f"This policy is expected to generate **{frequency:.4f} claims per year** "
            f"({frequency * 100:.2f}%). "
            f"The average frequency in the training dataset was **10.39%**. "
            f"This policy is **{frequency / 0.1039:.1f}x** the average."
        )

    except Exception as e:
        st.error(f"Endpoint error: {str(e)}")
        st.warning("Make sure the endpoint is deployed and running.")