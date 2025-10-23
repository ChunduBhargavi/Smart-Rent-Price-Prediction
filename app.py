import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime
import numpy as np

# --- Load model, encoders, scaler ---
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

# --- Categorical Inputs ---
encoder_keys = list(label_encoders.keys())
user_inputs = {}
for col in encoder_keys:
    classes = label_encoders[col].classes_
    user_inputs[col] = st.selectbox(col.replace("_", " ").title(), classes)

# --- Numeric Inputs ---
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

# Negotiable
user_inputs['Negotiable'] = 1 if st.checkbox("Negotiable", value=True) else 0

# Activation date
activation_date = st.date_input("Activation Date", value=datetime.today())
user_inputs['Year'] = activation_date.year
user_inputs['Month'] = activation_date.month
user_inputs['Day'] = activation_date.day

# --- Prepare input DataFrame ---
def prepare_input_df():
    df = pd.DataFrame([user_inputs])
    
    # Encode categoricals
    for col in encoder_keys:
        le = label_encoders[col]
        try:
            df[col] = le.transform(df[col])
        except:
            df[col] = 0  # fallback
    
    # Fill missing numeric columns
    for col in scaler.feature_names_in_:
        if col not in df.columns:
            df[col] = numeric_defaults.get(col, 0)
    
    # Scale numeric
    df[scaler.feature_names_in_] = scaler.transform(df[scaler.feature_names_in_])
    
    # Align with model columns
    for col in model.feature_names_in_:
        if col not in df.columns:
            df[col] = 0
    df = df[model.feature_names_in_]
    return df

# --- Predict Rent on button click ---
if st.button("Predict Rent"):
    input_df = prepare_input_df()
    try:
        # Predict (model trained on log1p(rent))
        pred_log1p = model.predict(input_df)
        
        # Convert back from log1p
        pred_rent = np.expm1(pred_log1p)
        
        # Clip to minimum rent (avoid 0 or negative)
        pred_rent = np.maximum(pred_rent, 10000)  # Minimum rent ₹100
        
        # Round for display
        pred_value = round(pred_rent[0])
        
        st.success(f"Predicted Monthly Rent: ₹{pred_value:,}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
