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
st.write("Enter property details and click 'Predict Rent'.")

# --- Categorical inputs ---
encoder_keys = list(label_encoders.keys())
user_inputs = {}

for col in encoder_keys:
    classes = label_encoders[col].classes_
    user_inputs[col] = st.selectbox(col.replace("_", " ").title(), classes)

# --- Numeric inputs ---
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

# --- Prepare input DataFrame ---
def prepare_input_df():
    df = pd.DataFrame([user_inputs])
    
    # Encode categorical columns
    for col in encoder_keys:
        le = label_encoders[col]
        try:
            df[col] = le.transform(df[col])
        except:
            df[col] = 0  # fallback for unseen categories
    
    # Fill missing numeric columns
    for col in scaler.feature_names_in_:
        if col not in df.columns:
            df[col] = numeric_defaults.get(col, 0)
    
    # Scale numeric columns
    df[scaler.feature_names_in_] = scaler.transform(df[scaler.feature_names_in_])
    
    # Ensure all model columns exist
    for col in model.feature_names_in_:
        if col not in df.columns:
            df[col] = 0
    df = df[model.feature_names_in_]
    
    return df

# --- Predict rent on button click ---
if st.button("Predict Rent"):
    input_df = prepare_input_df()
    try:
        pred_log = model.predict(input_df)
        
        # --- Inverse transformation ---
        # Use log1p if model was trained with np.log1p
        pred_rent = np.expm1(pred_log)  # OR np.exp(pred_log) if log(y)
        
        # Prevent negative values
        pred_rent = np.maximum(pred_rent, 0)
        
        pred_value = round(pred_rent[0])
        st.success(f"Predicted Monthly Rent: ₹{pred_value:,}")
        
    except Exception as e:
        st.error(f"Prediction failed: {e}")
