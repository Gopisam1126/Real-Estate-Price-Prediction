import streamlit as st
import pandas as pd
import pickle
import os
from dotenv import load_dotenv

load_dotenv()

path_to_model = os.getenv("PATH_TO_MODEL")

try:
    with open(path_to_model, "rb") as f:
        pipeline = pickle.load(f)
    required_keys = {"scaler", "model", "columns"}
    missing = required_keys - set(pipeline.keys())
    if missing:
        st.error(f"Loaded pipeline missing keys: {', '.join(missing)}")
        st.stop()
except FileNotFoundError:
    st.error(f"Model file not found at: {path_to_model}")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

st.title("🏠 Bengaluru House Price Predictor")

location = st.text_input("Enter the location:")
sqft     = st.number_input("Enter the total square feet:", min_value=300.0, max_value=16000.0, value=1000.0)
bath     = st.selectbox("Number of bathrooms:", range(1, 6))
bhk      = st.selectbox("Size (in BHK):", range(1, 10))

if st.button("Predict Price"):
    if not location.strip():
        st.warning("Please enter a valid location before predicting.")
    else:
        df_input = pd.DataFrame([{  
            "location": location.lower().strip(),
            "total_sqft": sqft,
            "bath":       bath,
            "size":       bhk
        }])

        try:
            df_encoded = pd.get_dummies(df_input)
            df_encoded = df_encoded.reindex(columns=pipeline["columns"], fill_value=0)
        except Exception as e:
            st.error(f"Error encoding input data: {e}")
            st.stop()

        try:
            X_scaled = pipeline["scaler"].transform(df_encoded)
            prediction = pipeline["model"].predict(X_scaled)[0]
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

        price = max(0.0, prediction)
        st.success(f"Predicted Price: ₹{price:,.2f}")
