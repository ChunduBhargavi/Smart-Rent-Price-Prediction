import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime

# --- Load model and encoder ---
@st.cache_resource
def load_model_and_encoder():
    with open('regression_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('ordinal_encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
    return model, encoder

model, encoder = load_model_and_encoder()

# --- Predefined categories ---
type_options = ['BHK1', 'BHK2', 'BHK3', 'BHK4', 'BHK4PLUS', 'RK1', 'PENTHOUSE']
lease_type_options = ['ANYONE', 'FAMILY', 'BACHELOR', 'COMPANY']
furnishing_options = ['SEMI_FURNISHED', 'FULLY_FURNISHED', 'NOT_FURNISHED']
parking_options = ['BOTH', 'BIKE', 'CAR', 'NONE']
facing_options = ['E', 'NE', 'N', 'SE', 'NW', 'S', 'W', 'SW']
water_supply_options = ['CORP_BORE', 'CORPORATION', 'BOREWELL']
building_type_options = ['AP', 'IF', 'IH', 'GC']

# --- Amenities list ---
amenities_list = ['LIFT', 'GYM', 'INTERNET', 'AC', 'CLUB', 'INTERCOM', 'POOL', 
                  'CPA', 'FS', 'SERVANT', 'SECURITY', 'SC', 'GP', 'PARK', 'RWH', 
                  'STP', 'HK', 'PB', 'VP']

# --- Streamlit app ---
st.title("Rental Property Price Predictor")

with st.form(key='rental_form'):
    # Date
    activation_date = st.date_input("Activation Date", value=datetime.today())
    
    # Location
    latitude = st.number_input("Latitude", value=12.93, format="%.6f")
    longitude = st.number_input("Longitude", value=77.67, format="%.6f")
    
    # Categorical
    type_ = st.selectbox("Type", type_options)
    lease_type = st.selectbox("Lease Type", lease_type_options)
    furnishing = st.selectbox("Furnishing", furnishing_options)
    parking = st.selectbox("Parking", parking_options)
    facing = st.selectbox("Facing", facing_options)
    water_supply = st.selectbox("Water Supply", water_supply_options)
    building_type = st.selectbox("Building Type", building_type_options)
    
    # Numerical
    property_size = st.number_input("Property Size (sq ft)", min_value=100, value=1400)
    property_age = st.number_input("Property Age (years)", min_value=0, value=4)
    bathroom = st.number_input("Bathrooms", min_value=1, value=2)
    cup_board = st.number_input("Cupboards", min_value=0, value=2)
    floor = st.number_input("Floor", min_value=0, value=3)
    total_floor = st.number_input("Total Floors", min_value=1, value=4)
    balconies = st.number_input("Balconies", min_value=0, value=2)
    negotiable = st.checkbox("Negotiable", value=True)
    
    # Amenities
    st.subheader("Amenities")
    amenities_dict = {}
    for amen in amenities_list:
        amenities_dict[amen] = 1 if st.checkbox(amen, value=False) else 0
    
    submit = st.form_submit_button("Predict Rental Price")

if submit:
    # Extract date parts
    year = activation_date.year
    month = activation_date.month
    day = activation_date.day
    
    # --- Build user DataFrame ---
    user_dict = {
        'month': month,
        'day': day,
        'year': year,
        'latitude': latitude,
        'longitude': longitude,
        'type': type_,
        'lease_type': lease_type,
        'furnishing': furnishing,
        'parking': parking,
        'facing': facing,
        'water_supply': water_supply,
        'building_type': building_type,
        'property_size': property_size,
        'property_age': property_age,
        'bathroom': bathroom,
        'cup_board': cup_board,
        'floor': floor,
        'total_floor': total_floor,
        'balconies': balconies,
        'negotiable': 1 if negotiable else 0,
        # main amenities
        'gym': amenities_dict['GYM'],
        'lift': amenities_dict['LIFT'],
        'swimming_pool': amenities_dict['POOL']
    }
    
    user_df = pd.DataFrame([user_dict])
    
    # --- Encode categorical features safely ---
    cat_cols = ['type', 'lease_type', 'furnishing', 'parking', 'facing', 'water_supply', 'building_type']
    for col in cat_cols:
        try:
            user_df[[col]] = encoder.transform(user_df[[col]])
        except ValueError:
            user_df[col] = -1  # fallback for unseen category
    
    # --- Add remaining amenities avoiding duplicates ---
    for amen in amenities_list:
        col_name = amen.lower()
        if col_name not in ['gym', 'lift', 'swimming_pool']:
            user_df[col_name] = amenities_dict[amen]
    
    # --- Ensure columns match model ---
    model_cols = model.feature_names_in_
    for col in model_cols:
        if col not in user_df.columns:
            user_df[col] = 0  # default value
    
    user_df = user_df[model_cols]  # reorder
    
    # --- Predict ---
    try:
        prediction = model.predict(user_df)
        st.success(f"Predicted Rental Price: ₹{prediction[0]:,.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
