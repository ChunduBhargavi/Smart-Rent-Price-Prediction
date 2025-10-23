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
st.write("Enter property details to predict monthly rent.")

# --- Categorical inputs ---
encoder_keys = list(label_encoders.keys())
user_inputs = {}

for col in encoder_keys:
    classes = label_encoders[col].classes_
    user_inputs[col] = st.selectbox(col.replace("_", " ").title(), classes)

# --- Numeric inputs with defaults ---
numeric_defaults = {
    'BHK': 2,
    'Size': 1000,
    'Bathroom': 2,
    'Property Age': 4,
    'Cupboard': 2,
    'Floor': 3,
    'Total Floor': 4,
    'Balconies': 2
}

for col, default in numeric_defaults.items():
    user_inputs[col] = st.number_input(col.replace("_", " "), min_value=0, value=default)

# Negotiable checkbox
user_inputs['Negotiable'] = 1 if st.checkbox("Negotiable", value=True) else 0

# Activation date
activation_date = st.date_input("Activation Date", value=datetime.today())
user_inputs['Year'] = activation_date.year
user_inputs['Month'] = activation_date.month
user_inputs['Day'] = activation_date.day

# --- Amenities ---
all_amenities = ['LIFT', 'GYM', 'INTERNET', 'AC', 'CLUB', 'INTERCOM', 'POOL',
                 'CPA', 'FS', 'SERVANT', 'SECURITY', 'SC', 'GP', 'PARK', 'RWH',
                 'STP', 'HK', 'PB', 'VP']

st.subheader("Amenities")
selected_amenities = {}
for amen in all_amenities:
    selected_amenities[amen] = 1 if st.checkbox(amen, value=False) else 0

# --- Prepare DataFrame ---
input_df = pd.DataFrame([user_inputs])

# Only include amenities that exist in model features
model_amenities = [a for a in all_amenities if a in model.feature_names_in_]
amenities_df = pd.DataFrame([{k: v for k, v in selected_amenities.items() if k in model_amenities}])
input_df = pd.concat([input_df, amenities_df], axis=1)

# --- Encode categorical variables ---
for col in encoder_keys:
    le = label_encoders[col]
    input_df[col] = le.transform(input_df[col])

# --- Scale numeric columns ---
for col in scaler.feature_names_in_:
    if col not in input_df.columns:
        input_df[col] = numeric_defaults.get(col, 0)
input_df[scaler.feature_names_in_] = scaler.transform(input_df[scaler.feature_names_in_])

# --- Align input columns with model ---
for col in model.feature_names_in_:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[model.feature_names_in_]

# --- Predict Rent ---
if st.button("Predict Rent"):
    try:
        pred = model.predict(input_df)
        
        # Handle log-transform if needed
        if hasattr(model, 'log_transform') and model.log_transform:
            pred = np.exp(pred)
        
        # Clamp very low predictions
        pred = [max(1000, p) for p in pred]
        
        st.success(f"Predicted Monthly Rent: ₹{pred[0]:,.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
