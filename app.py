import streamlit as st
import pandas as pd
import joblib

# Load model and preprocessing objects
model = joblib.load("final_rent_model.joblib")
label_encoders = joblib.load("label_encoders.joblib")
scaler = joblib.load("scaler.joblib")

st.title("Smart Rent Price Prediction App")
st.write("Enter property details to predict monthly rent")

# Input fields
city = st.text_input("City Name", "Bangalore")
area_type = st.selectbox("Area Type", ["Super Area", "Carpet Area", "Built Area"])
bhk = st.number_input("BHK (Bedrooms)", min_value=1, max_value=10, value=2)
size = st.number_input("Size (in sq.ft)", min_value=100, max_value=5000, value=1000)
furnishing = st.selectbox("Furnishing Status", ["Furnished", "Semi-Furnished", "Unfurnished"])
tenant = st.selectbox("Tenant Preferred", ["Bachelors", "Family", "Company"])
bathroom = st.number_input("Number of Bathrooms", min_value=1, max_value=5, value=2)

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

# Encode categorical columns
for col, le in label_encoders.items():
    if col in input_df.columns:
        try:
            input_df[col] = le.transform(input_df[col])
        except:
            input_df[col] = 0  # default for unseen values

# Scale numerical columns
num_cols = ['Size', 'BHK', 'Bathroom']
input_df[num_cols] = scaler.transform(input_df[num_cols])

# Predict rent
if st.button("Predict Rent"):
    pred = model.predict(input_df)
    st.success(f"Predicted Monthly Rent: ₹{pred[0]:,.2f}")
