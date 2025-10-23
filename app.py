import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime

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
activation_date = st.date_input("Activation Date", value=datetime.today())

# --- Categorical inputs only from known encoder categories ---
city = st.selectbox("City Name", label_encoders['City'].classes_)
area_type = st.selectbox("Area Type", label_encoders['Area Type'].classes_)
furnishing = st.selectbox("Furnishing Status", label_encoders['Furnishing Status'].classes_)
tenant = st.selectbox("Tenant Preferred", label_encoders['Tenant Preferred'].classes_)

bhk = st.number_input("BHK (Bedrooms)", 1, 10, 2)
size = st.number_input("Size (sq.ft)", 100, 5000, 1000)
bathroom = st.number_input("Number of Bathrooms", 1, 5, 2)

# Other numeric inputs
property_age = st.number_input("Property Age (years)", min_value=0, value=4)
cup_board = st.number_input("Cupboards", min_value=0, value=2)
floor = st.number_input("Floor", min_value=0, value=3)
total_floor = st.number_input("Total Floors", min_value=1, value=4)
balconies = st.number_input("Balconies", min_value=0, value=2)
negotiable = st.checkbox("Negotiable", value=True)

# Amenities
amenities_list = ['LIFT', 'GYM', 'INTERNET', 'AC', 'CLUB', 'INTERCOM', 'POOL',
                  'CPA', 'FS', 'SERVANT', 'SECURITY', 'SC', 'GP', 'PARK', 'RWH',
                  'STP', 'HK', 'PB', 'VP']

st.subheader("Amenities")
amenities_dict = {}
for amen in amenities_list:
    amenities_dict[amen] = 1 if st.checkbox(amen, value=False) else 0

# --- Prepare input DataFrame ---
year = activation_date.year
month = activation_date.month
day = activation_date.day

input_dict = {
    'City': city,
    'Area Type': area_type,
    'BHK': bhk,
    'Size': size,
    'Furnishing Status': furnishing,
    'Tenant Preferred': tenant,
    'Bathroom': bathroom,
    'Year': year,
    'Month': month,
    'Day': day,
    'Property Age': property_age,
    'Cupboard': cup_board,
    'Floor': floor,
    'Total Floor': total_floor,
    'Balconies': balconies,
    'Negotiable': 1 if negotiable else 0,
    # main amenities
    'Gym': amenities_dict['GYM'],
    'Lift': amenities_dict['LIFT'],
    'Swimming Pool': amenities_dict['POOL']
}

input_df = pd.DataFrame([input_dict])

# --- Encode categorical features ---
cat_cols = ['City', 'Area Type', 'Furnishing Status', 'Tenant Preferred']
for col in cat_cols:
    le = label_encoders[col]
    input_df[col] = le.transform(input_df[col])

# --- Scale numeric columns ---
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
        st.success(f"Predicted Monthly Rent: ₹{pred[0]:,.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
