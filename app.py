import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ─── Page Config ───────────────────────────────────────────
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# ─── Load Model ────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("house_model.pkl")

model = load_model()

# ─── Header ────────────────────────────────────────────────
st.title("🏠 House Price Predictor")    
st.caption("India Real Estate — ML Powered Price Estimation")
st.divider()

# ─── Input Form ────────────────────────────────────────────
st.subheader("🏡 Property Details")

col1, col2 = st.columns(2)

with col1:
    area = st.number_input("Area (sq ft)", min_value=1000, max_value=16200, value=5000, step=100)
    bedrooms = st.selectbox("Bedrooms", options=[1, 2, 3, 4, 5, 6], index=2)
    bathrooms = st.selectbox("Bathrooms", options=[1, 2, 3, 4], index=1)
    stories = st.selectbox("Stories (Floors)", options=[1, 2, 3, 4], index=1)
    parking = st.selectbox("Parking Spaces", options=[0, 1, 2, 3], index=1)

with col2:
    mainroad = st.selectbox("Main Road Access", options=["yes", "no"], index=0)
    guestroom = st.selectbox("Guest Room", options=["yes", "no"], index=1)
    basement = st.selectbox("Basement", options=["yes", "no"], index=1)
    hotwaterheating = st.selectbox("Hot Water Heating", options=["yes", "no"], index=1)
    airconditioning = st.selectbox("Air Conditioning", options=["yes", "no"], index=0)

col3, col4 = st.columns(2)
with col3:
    prefarea = st.selectbox("Preferred Area", options=["yes", "no"], index=0)
with col4:
    furnishingstatus = st.selectbox(
        "Furnishing Status",
        options=["furnished", "semi-furnished", "unfurnished"],
        index=0
    )

st.divider()

# ─── Prediction ────────────────────────────────────────────
if st.button("🔮 Predict Price", use_container_width=True, type="primary"):

    input_df = pd.DataFrame({
        "area": [area],
        "bedrooms": [bedrooms],
        "bathrooms": [bathrooms],
        "stories": [stories],
        "mainroad": [mainroad],
        "guestroom": [guestroom],
        "basement": [basement],
        "hotwaterheating": [hotwaterheating],
        "airconditioning": [airconditioning],
        "parking": [parking],
        "prefarea": [prefarea],
        "furnishingstatus": [furnishingstatus]
    })

    predicted_price = model.predict(input_df)[0]

    # ─── Result Display ─────────────────────────────────
    st.success("✅ Prediction Complete!")

    col_r1, col_r2, col_r3 = st.columns(3)

    with col_r1:
        st.metric(
            label="Predicted Price",
            value=f"₹{predicted_price:,.0f}"
        )
    with col_r2:
        st.metric(
            label="In Lakhs",
            value=f"₹{predicted_price/100000:.2f}L"
        )
    with col_r3:
        st.metric(
            label="In Crores",
            value=f"₹{predicted_price/10000000:.3f}Cr"
        )

    st.divider()

    # ─── Input Summary ──────────────────────────────────
    with st.expander("📋 Input Summary"):
        st.dataframe(input_df, use_container_width=True)

    # ─── Model Info ─────────────────────────────────────
    with st.expander("📊 Model Performance"):
        m1, m2, m3 = st.columns(3)
        m1.metric("Best Model", "Linear Regression")
        m2.metric("R2 Score", "0.65")
        m3.metric("Dataset", "545 houses")

    st.info("ℹ️ Prices are in Indian Rupees (₹). Dataset contains real Indian housing data.")