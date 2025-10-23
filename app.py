import streamlit as st
import pandas as pd
import joblib

# Load saved objects
model = joblib.load("final_rent_model.joblib")
label_encoders = joblib.load("label_encoders.joblib")
scaler = joblib.load("scaler.joblib")

st.title("Smart Rent Price Prediction")
st.write("Enter property details below to predict the monthly rent.")

# User inputs
city = st.text_input("City Name", "Bangalore")
area_type = st.selectbox("Area Type", ["Super Area", "Carpet Area", "Built Area"])
bhk = st.number_input("BHK (Bedrooms)", 1, 10, 2)
size = st.number_input("Size (sq.ft)", 100, 5000, 1000)
furnishing = st.selectbox("Furnishing Status", ["Furnished", "Semi-Furnished", "Unfurnished"])
tenant = st.selectbox("Tenant Preferred", ["Bachelors", "Family", "Company"])
bathroom = st.number_input("Number of Bathrooms", 1, 5, 2)

# Create input DataFrame
input_df = pd.DataFrame({
    'City': [city],
    'Area Type': [area_type],
    'BHK': [bhk],
    'Size': [size],
    'Furnishing Status': [furnishing],
    'Tenant Preferred': [tenant],
    'Bathroom': [bathroom]
})

# Apply label encoding safely
for col, le in label_encoders.items():
    if col in input_df.columns:
        try:
            input_df[col] = le.transform(input_df[col])
        except ValueError:
            # If new/unseen label, assign 0
            input_df[col] = 0

# Ensure all numeric columns expected by the scaler are present
num_cols = scaler.feature_names_in_  # numeric columns scaler was fitted on
for col in num_cols:
    if col not in input_df.columns:
        input_df[col] = 0  # default value

# Reorder columns to match scaler
input_df = input_df[num_cols]

# Scale numeric features
input_df[num_cols] = scaler.transform(input_df[num_cols])

# Predict button
if st.button("Predict Rent"):
    pred = model.predict(input_df)
    st.success(f"Predicted Monthly Rent: ₹{pred[0]:,.2f}")


