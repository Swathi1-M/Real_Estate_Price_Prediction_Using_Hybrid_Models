# streamlit_app.py

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from PIL import Image

# Load saved models and scaler
xgb_model = joblib.load("models/xgboost_model.pkl")
ann_model = tf.keras.models.load_model("models/ann_model.keras", compile=False)
meta_model = joblib.load("models/meta_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Set Streamlit page layout
st.set_page_config(page_title="Real Estate Price Predictor", layout="centered")

# Title and Instructions
st.title("🏠 Real Estate Price Prediction")
st.markdown("""
This app uses a **Hybrid AI Model (XGBoost + ANN + Meta Learner)** to estimate house sale prices.

👉 Fill in the property details below and click **Predict** to get an estimated sale price.
""")

# Input form
col1, col2 = st.columns(2)
with col1:
    grlivarea = st.number_input("Above Ground Living Area (sqft)", value=1500)
    yearbuilt = st.number_input("Year Built", value=2000)
with col2:
    overallqual = st.slider("Overall Quality (1-10)", 1, 10, 5)
    garagecars = st.slider("Garage Capacity (Cars)", 0, 4, 2)

totalbsmt = st.number_input("Total Basement Area (sqft)", value=800)

# Predict button
if st.button("Predict Price"):
    try:
        input_data = np.array([[grlivarea, overallqual, yearbuilt, garagecars, totalbsmt]])
        input_scaled = scaler.transform(input_data)

        # Individual model predictions
        xgb_pred = xgb_model.predict(input_scaled)
        ann_pred = ann_model.predict(input_scaled)

        # Combine predictions
        meta_input = np.column_stack((xgb_pred, ann_pred.flatten()))
        final_pred = meta_model.predict(meta_input)

        st.success(f"🏡 Estimated House Price: **${final_pred[0]:,.2f}**")
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

# Footer
st.markdown("---")
st.caption("Built with ❤️ using Streamlit, XGBoost, and Keras")
