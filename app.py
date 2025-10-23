import streamlit as st
import pandas as pd
import joblib

# Load saved objects
model = joblib.load("final_rent_model.joblib")
label_encoders = joblib.load("label_encoders.joblib")
scaler = joblib.load("scaler.joblib")

st.title("Smart Rent Price Prediction")
st.write("Enter property details below to predict the monthly rent.")

# --- User Inputs ---
city = st.text_input("City Name", "Bangalore")
area_type = st.selectbox("Area Type", ["Super Area", "Carpet Area", "Built Area"])
bhk = st.number_input("BHK (Bedrooms)", 1, 10, 2)
size = st.number_input("Size (sq.ft)", 100, 5000, 1000)
furnishing = st.selectbox("Furnishing Status", ["Furnished", "Semi-Furnished", "Unfurnished"])
tenant = st.selectbox("Tenant Preferred", ["Bachelors", "Family", "Company"])
bathroom = st.number_input("Number of Bathrooms", 1, 5, 2)

# --- Create Input DataFrame ---
input_df = pd.DataFrame({
    'City': [city],
    'Area Type': [area_type],
    'BHK': [bhk],
    'Size': [size],
    'Furnishing Status': [furnishing],
    'Tenant Preferred': [tenant],
    'Bathroom': [bathroom]
})

# --- Encode Categorical Features Safely ---
for col, le in label_encoders.items():
    if col in input_df.columns:
        try:
            input_df[col] = le.transform(input_df[col])
        except ValueError:
            # Assign 0 if unseen category
            input_df[col] = 0

# --- Scale Numeric Features ---
num_cols = scaler.feature_names_in_
for col in num_cols:
    if col not in input_df.columns:
        input_df[col] = 0  # fallback for missing numeric columns

# Ensure column order matches scaler
input_df[num_cols] = scaler.transform(input_df[num_cols])

# --- Ensure Columns Match Model Features ---
model_cols = model.feature_names_in_
for col in model_cols:
    if col not in input_df.columns:
        input_df[col] = 0  # fallback for missing features

# Reorder columns exactly as model expects
input_df = input_df[model_cols]

# --- Predict ---
if st.button("Predict Rent"):
    try:
        pred = model.predict(input_df)
        st.success(f"Predicted Monthly Rent: ₹{pred[0]:,.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")


