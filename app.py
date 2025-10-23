import streamlit as st
import pandas as pd
import joblib

model = joblib.load("final_rent_model.joblib")
label_encoders = joblib.load("label_encoders.joblib")
scaler = joblib.load("scaler.joblib")

st.title("Smart Rent Price Prediction")
st.write("Enter property details below to predict the monthly rent.")

city = st.text_input("City Name", "Bangalore")
area_type = st.selectbox("Area Type", ["Super Area", "Carpet Area", "Built Area"])
bhk = st.number_input("BHK (Bedrooms)", 1, 10, 2)
size = st.number_input("Size (sq.ft)", 100, 5000, 1000)
furnishing = st.selectbox("Furnishing Status", ["Furnished", "Semi-Furnished", "Unfurnished"])
tenant = st.selectbox("Tenant Preferred", ["Bachelors", "Family", "Company"])
bathroom = st.number_input("Number of Bathrooms", 1, 5, 2)

input_df = pd.DataFrame({
    'City': [city],
    'Area Type': [area_type],
    'BHK': [bhk],
    'Size': [size],
    'Furnishing Status': [furnishing],
    'Tenant Preferred': [tenant],
    'Bathroom': [bathroom]
})

for col, le in label_encoders.items():
    if col in input_df.columns:
        try:
            input_df[col] = le.transform(input_df[col])
        except:
            input_df[col] = 0

num_cols = ['Size', 'BHK', 'Bathroom']
input_df[num_cols] = scaler.transform(input_df[num_cols])

if st.button("Predict Rent"):
    pred = model.predict(input_df)
    st.success(f"Predicted Monthly Rent: ₹{pred[0]:,.2f}")

