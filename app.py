import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import datetime

# -------------------------
# Load model, encoder, scaler
# -------------------------
@st.cache_resource
def load_model_objects():
    model = joblib.load("final_rent_model.joblib")
    encoder = joblib.load("label_encoders.joblib")
    scaler = joblib.load("scaler.joblib")
    return model, encoder, scaler

model, encoder, scaler = load_model_objects()

# -------------------------
# Streamlit UI
# -------------------------
st.title("Smart Rent Price Prediction")
st.write("Enter property details and click 'Predict Rent'.")

# Categorical columns
cat_cols = list(encoder.keys())

# Numeric defaults
numeric_defaults = {
    'BHK': 2,
    'Size': 1000,
    'Bathroom': 2,
    'Property_Age': 4,
    'Cupboard': 2,
    'Floor': 3,
    'Total_Floor': 4,
    'Balconies': 2
}

# --- User Inputs ---
with st.form(key='rental_form'):
    user_inputs = {}
    
    # Categorical
    for col in cat_cols:
        user_inputs[col] = st.selectbox(col.replace("_", " ").title(), encoder[col].classes_)
    
    # Numeric
    for col, default in numeric_defaults.items():
        user_inputs[col] = st.number_input(col.replace("_", " "), min_value=0, value=default)
    
    # Negotiable
    user_inputs['Negotiable'] = 1 if st.checkbox("Negotiable", value=True) else 0
    
    # Activation date
    activation_date = st.date_input("Activation Date", value=datetime.today())
    user_inputs['Year'] = activation_date.year
    user_inputs['Month'] = activation_date.month
    user_inputs['Day'] = activation_date.day
    
    submit = st.form_submit_button("Predict Rent")

# -------------------------
# Prepare input DataFrame and predict
# -------------------------
def prepare_input_df(inputs):
    df = pd.DataFrame([inputs])
    
    # Encode categorical columns
    for col in cat_cols:
        try:
            df[col] = encoder[col].transform(df[col])
        except:
            df[col] = 0  # fallback for unseen category
    
    # Ensure numeric columns expected by scaler exist
    num_cols = scaler.feature_names_in_
    for col in num_cols:
        if col not in df.columns:
            df[col] = 0
    
    # Scale numeric features
    df[num_cols] = scaler.transform(df[num_cols])
    
    # Align columns with model
    for col in model.feature_names_in_:
        if col not in df.columns:
            df[col] = 0
    df = df[model.feature_names_in_]
    
    return df

if submit:
    input_df = prepare_input_df(user_inputs)
    
    try:
        pred_log1p = model.predict(input_df)
        # Convert from log1p if model uses it
        pred_rent = np.expm1(pred_log1p)
        pred_rent = np.clip(pred_rent, 0, None)  # avoid negative/zero
        
        st.success(f"Predicted Monthly Rent: ₹{round(pred_rent[0]):,}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
