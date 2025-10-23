import streamlit as st
import pandas as pd
import joblib
import os

# --- Load model, label encoders, and scaler ---
@st.cache_resource
def load_model_objects():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    model_path = os.path.join(BASE_DIR, 'final_rent_model.joblib')
    encoder_path = os.path.join(BASE_DIR, 'label_encoders.joblib')
    scaler_path = os.path.join(BASE_DIR, 'scaler.joblib')
    
    model = joblib.load(model_path)
    label_encoders = joblib.load(encoder_path)
    scaler = joblib.load(scaler_path)
    
    return model, label_encoders, scaler

model, label_encoders, scaler = load_model_objects()

# --- Streamlit UI ---
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

# Encode categorical columns safely
for col, le in label_encoders.items():
    if col in input_df.columns:
        try:
            input_df[col] = le.transform(input_df[col])
        except ValueError:
            input_df[col] = 0  # fallback for unseen categories

# Scale numeric columns
num_cols = scaler.feature_names_in_
for col in num_cols:
    if col not in input_df.columns:
        input_df[col] = 0
input_df[num_cols] = scaler.transform(input_df[num_cols])

# Ensure columns match model
model_cols = model.feature_names_in_
for col in model_cols:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[model_cols]  # reorder

# Predict
if st.button("Predict Rent"):
    try:
        pred = model.predict(input_df)
        st.success(f"Predicted Monthly Rent: ₹{pred[0]:,.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
