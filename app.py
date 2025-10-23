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
st.write("Enter property details to predict monthly rent dynamically.")

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

# --- Prepare DataFrame for prediction ---
input_df = pd.DataFrame([user_inputs])

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

# --- Dynamic Prediction ---
# Using a function so prediction updates whenever inputs change
def predict_rent(df):
    try:
        pred = model.predict(df)
        # Handle log-transform if model was trained on log(rent)
        if hasattr(model, 'log_transform') and model.log_transform:
            pred = np.exp(pred)
        # Clamp very low predictions
        pred = [max(1000, p) for p in pred]
        return pred[0]
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return None

predicted_rent = predict_rent(input_df)
if predicted_rent is not None:
    st.success(f"Predicted Monthly Rent: ₹{predicted_rent:,.2f}")
