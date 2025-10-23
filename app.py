import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime
import numpy as np

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

# --- Identify actual encoder keys ---
encoder_keys = list(label_encoders.keys())

# --- User inputs ---
activation_date = st.date_input("Activation Date", value=datetime.today())

# Select categorical options from encoder classes
user_inputs = {}
for col in encoder_keys:
    classes = label_encoders[col].classes_
    user_inputs[col] = st.selectbox(col.replace("_", " ").title(), classes)

# Numeric inputs
numeric_cols_defaults = {
    'BHK': 2,
    'Size': 1000,
    'Bathroom': 2,
    'Property Age': 4,
    'Cupboard': 2,
    'Floor': 3,
    'Total Floor': 4,
    'Balconies': 2
}

for col, default in numeric_cols_defaults.items():
    user_inputs[col] = st.number_input(col.replace("_", " "), min_value=0, value=default)

# Negotiable
user_inputs['Negotiable'] = 1 if st.checkbox("Negotiable", value=True) else 0

# Amenities
amenities_list = ['LIFT', 'GYM', 'INTERNET', 'AC', 'CLUB', 'INTERCOM', 'POOL',
                  'CPA', 'FS', 'SERVANT', 'SECURITY', 'SC', 'GP', 'PARK', 'RWH',
                  'STP', 'HK', 'PB', 'VP']

st.subheader("Amenities")
for amen in amenities_list:
    user_inputs[amen] = 1 if st.checkbox(amen, value=False) else 0

# --- Prepare DataFrame ---
year = activation_date.year
month = activation_date.month
day = activation_date.day

user_inputs['Year'] = year
user_inputs['Month'] = month
user_inputs['Day'] = day

input_df = pd.DataFrame([user_inputs])

# --- Encode categorical features ---
for col in encoder_keys:
    le = label_encoders[col]
    input_df[col] = le.transform(input_df[col])

# --- Scale numeric features ---
num_cols = scaler.feature_names_in_
for col in num_cols:
    if col not in input_df.columns:
        input_df[col] = 0
input_df[num_cols] = scaler.transform(input_df[num_cols])

# --- Ensure model columns order ---
model_cols = model.feature_names_in_
for col in model_cols:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[model_cols]

# --- Predict ---
if st.button("Predict Rent"):
    try:
        pred = model.predict(input_df)
        
        # --- Handle log-transform if model trained on log(rent) ---
        if hasattr(model, 'log_transform') and model.log_transform:
            pred = np.exp(pred)  # convert back to normal scale
        
        # --- Clamp negative predictions to 0 ---
        pred = [max(0, p) for p in pred]
        
        st.success(f"Predicted Monthly Rent: ₹{pred[0]:,.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
