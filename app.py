import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import datetime

# --- Load model and encoder ---
@st.cache_resource
def load_model_objects():
    model = joblib.load("final_rent_model.joblib")
    encoder = joblib.load("label_encoders.joblib")
    scaler = joblib.load("scaler.joblib")
    return model, encoder, scaler

model, encoder, scaler = load_model_objects()

# --- Streamlit UI ---
st.title("Smart Rent Price Prediction")
st.write("Enter property details and click 'Predict Rent'.")

# --- Form Inputs ---
with st.form(key='rental_form'):
    # Date
    activation_date = st.date_input("Activation Date", value=datetime.today())
    
    # Categorical
    type_ = st.selectbox("Type", encoder['type'].classes_)
    lease_type = st.selectbox("Lease Type", encoder['lease_type'].classes_)
    furnishing = st.selectbox("Furnishing", encoder['furnishing'].classes_)
    parking = st.selectbox("Parking", encoder['parking'].classes_)
    facing = st.selectbox("Facing", encoder['facing'].classes_)
    water_supply = st.selectbox("Water Supply", encoder['water_supply'].classes_)
    building_type = st.selectbox("Building Type", encoder['building_type'].classes_)
    
    # Numeric
    property_size = st.number_input("Property Size (sq ft)", min_value=100, value=1400)
    bhk = st.number_input("BHK (Bedrooms)", min_value=1, value=2)
    bathroom = st.number_input("Number of Bathrooms", min_value=1, value=2)
    property_age = st.number_input("Property Age (years)", min_value=0, value=4)
    cup_board = st.number_input("Cupboards", min_value=0, value=2)
    floor = st.number_input("Floor", min_value=0, value=3)
    total_floor = st.number_input("Total Floors", min_value=1, value=4)
    balconies = st.number_input("Balconies", min_value=0, value=2)
    negotiable = st.checkbox("Negotiable", value=True)
    
    # Amenities used by model
    st.subheader("Amenities (affect prediction)")
    gym = st.checkbox("Gym", value=False)
    lift = st.checkbox("Lift", value=False)
    swimming_pool = st.checkbox("Swimming Pool", value=False)
    
    submit = st.form_submit_button("Predict Rent")

# --- Prepare DataFrame and Encode ---
if submit:
    year, month, day = activation_date.year, activation_date.month, activation_date.day
    
    user_dict = {
        'type': type_,
        'lease_type': lease_type,
        'furnishing': furnishing,
        'parking': parking,
        'facing': facing,
        'water_supply': water_supply,
        'building_type': building_type,
        'Size': property_size,
        'BHK': bhk,
        'Bathroom': bathroom,
        'Property_Age': property_age,
        'Cupboard': cup_board,
        'Floor': floor,
        'Total_Floor': total_floor,
        'Balconies': balconies,
        'Negotiable': 1 if negotiable else 0,
        'Year': year,
        'Month': month,
        'Day': day,
        'gym': int(gym),
        'lift': int(lift),
        'swimming_pool': int(swimming_pool)
    }
    
    input_df = pd.DataFrame([user_dict])
    
    # Encode categoricals
    cat_cols = ['type', 'lease_type', 'furnishing', 'parking', 'facing', 'water_supply', 'building_type']
    for col in cat_cols:
        input_df[col] = encoder[col].transform(input_df[col])
    
    # Scale numeric columns
    num_cols = scaler.feature_names_in_
    input_df[num_cols] = scaler.transform(input_df[num_cols])
    
    # Align columns with model
    for col in model.feature_names_in_:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[model.feature_names_in_]
    
    # --- Predict ---
    try:
        pred_log1p = model.predict(input_df)
        pred_rent = np.expm1(pred_log1p)  # convert back if model trained on log1p
        pred_rent = np.clip(pred_rent, 0, None)  # avoid negative values
        st.success(f"Predicted Monthly Rent: ₹{round(pred_rent[0]):,}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
