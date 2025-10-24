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
st.write("Enter property details and amenities below to estimate the monthly rent.")

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

# --- Amenities to include ---
amenities_list = [
    "Lift",
    "Car_Parking",
    "Power_Backup",
    "Security",
    "Gym",
    "Swimming_Pool",
    "Park",
    "Club_House",
    "Children_Play_Area",
    "Intercom"
]

# --- User Inputs ---
with st.form(key='rental_form'):
    user_inputs = {}
    
    st.subheader("Property Details")
    for col in cat_cols:
        user_inputs[col] = st.selectbox(col.replace("_", " ").title(), encoder[col].classes_)
    for col, default in numeric_defaults.items():
        user_inputs[col] = st.number_input(col.replace("_", " "), min_value=0, value=default)
    user_inputs['Negotiable'] = 1 if st.checkbox("Is the rent negotiable?", value=True) else 0
    
    st.subheader("Amenities")
    selected_amenities = []
    cols = st.columns(2)
    for i, amenity in enumerate(amenities_list):
        with cols[i % 2]:
            checked = st.checkbox(amenity.replace("_", " "))
            user_inputs[amenity] = 1 if checked else 0  # convert to numeric 1/0
            if checked:
                selected_amenities.append(amenity)
    
    st.subheader("Listing Date")
    activation_date = st.date_input("Activation Date", value=datetime.today())
    user_inputs['Year'] = activation_date.year
    user_inputs['Month'] = activation_date.month
    user_inputs['Day'] = activation_date.day
    
    submit = st.form_submit_button("🔮 Predict Rent")

# -------------------------
# Prepare input DataFrame
# -------------------------
def prepare_input_df(inputs):
    df = pd.DataFrame([inputs])
    
    # Encode categorical columns
    for col in cat_cols:
        try:
            df[col] = encoder[col].transform(df[col])
        except:
            df[col] = 0
    
    # Ensure all numeric columns exist
    for col in scaler.feature_names_in_:
        if col not in df.columns:
            df[col] = 0
    
    # Scale numeric columns
    df[scaler.feature_names_in_] = scaler.transform(df[scaler.feature_names_in_])
    
    # Align columns with model
    for col in model.feature_names_in_:
        if col not in df.columns:
            df[col] = 0
    df = df[model.feature_names_in_]
    
    return df

# -------------------------
# Prediction
# -------------------------
if submit:
    try:
        input_df = prepare_input_df(user_inputs)
        pred_log1p = model.predict(input_df)
        pred_rent = np.expm1(pred_log1p)

        if np.isnan(pred_rent[0]) or pred_rent[0] <= 0:
            st.warning("The predicted rent seems invalid. Please review your input details.")
        else:
            st.success(f"**Predicted Monthly Rent:** ₹{round(pred_rent[0]):,}")
            
            if selected_amenities:
                st.info(f"Amenities considered: {', '.join(selected_amenities)}")
            else:
                st.info("No amenities selected.")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
