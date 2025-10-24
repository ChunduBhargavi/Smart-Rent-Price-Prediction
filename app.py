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
st.write("Enter property details and select amenities to estimate the monthly rent.")

# Categorical columns (loaded from encoder)
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

# -------------------------
# Amenities List (Your JSON)
# -------------------------
amenities = {
    "LIFT": False,
    "GYM": False,
    "INTERNET": False,
    "AC": False,
    "CLUB": False,
    "INTERCOM": False,
    "POOL": False,
    "CPA": False,           # Car Parking Area
    "FS": False,            # Fire Safety
    "SERVANT": False,
    "SECURITY": False,
    "SC": False,            # Shopping Complex
    "GP": False,            # Gas Pipeline
    "PARK": False,
    "RWH": False,           # Rain Water Harvesting
    "STP": False,           # Sewage Treatment Plant
    "HK": False,            # House Keeping
    "PB": False,            # Power Backup
    "VP": False             # Visitor Parking
}

# --- User Inputs ---
with st.form(key='rental_form'):
    user_inputs = {}
    
    st.subheader("Property Details")
    for col in cat_cols:
        user_inputs[col] = st.selectbox(col.replace("_", " ").title(), encoder[col].classes_)

    for col, default in numeric_defaults.items():
        user_inputs[col] = st.number_input(col.replace("_", " "), min_value=0, value=default)
    
    st.subheader("Amenities")
    selected_amenities = []
    cols = st.columns(3)
    for i, amenity in enumerate(amenities.keys()):
        with cols[i % 3]:
            checked = st.checkbox(amenity, value=amenities[amenity])
            user_inputs[amenity] = 1 if checked else 0
            if checked:
                selected_amenities.append(amenity)

    user_inputs['Negotiable'] = 1 if st.checkbox("Negotiable", value=True) else 0

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
        # 1) Build the raw input DataFrame from user inputs
        raw_df = pd.DataFrame([user_inputs])

        # 2) Show raw inputs (categorical strings, numeric values, amenity booleans)
        with st.expander("Raw input (before encoding/scaling)", expanded=False):
            st.write(raw_df.T)

        # 3) Encode categorical columns (make a copy for diagnostics)
        enc_df = raw_df.copy()
        for col in cat_cols:
            try:
                enc_df[col] = encoder[col].transform(enc_df[col])
            except Exception as enc_err:
                # fallback for unseen category: set 0 and show a warning
                enc_df[col] = 0
                st.warning(f"Warning: unseen category found for '{col}'. Using 0 as fallback.")

        with st.expander("After encoding categorical columns", expanded=False):
            st.write(enc_df.T)

        # 4) Ensure all numeric columns expected by scaler exist
        num_cols = list(scaler.feature_names_in_)
        for col in num_cols:
            if col not in enc_df.columns:
                enc_df[col] = 0

        # 5) Show the features that scaler expects and model expects
        with st.expander("Model & Scaler feature lists", expanded=False):
            st.write("Scaler expects (n):", len(num_cols))
            st.write(num_cols)
            st.write("---")
            st.write("Model expects (n):", len(model.feature_names_in_))
            st.write(list(model.feature_names_in_))

        # 6) Scale numeric columns (catch possible errors)
        try:
            enc_df[num_cols] = scaler.transform(enc_df[num_cols])
        except Exception as e_scaler:
            st.error(f"Scaler.transform failed: {e_scaler}")
            st.stop()

        with st.expander("After scaling (first 20 features shown)", expanded=False):
            st.write(enc_df.loc[:, enc_df.columns.isin(model.feature_names_in_)].T.head(20))

        # 7) Align columns with model
        for col in model.feature_names_in_:
            if col not in enc_df.columns:
                enc_df[col] = 0
        model_input = enc_df[model.feature_names_in_]

        with st.expander("Final model input (shape {})".format(model_input.shape), expanded=True):
            st.write(model_input.T)

        # 8) Predict and show raw outputs
        raw_pred = model.predict(model_input)  # raw model output
        st.write("Raw model output:", raw_pred)

        # 9) Convert if model used log1p (wrap in try in case not)
        try:
            pred_rent = np.expm1(raw_pred)
            converted = True
        except Exception:
            pred_rent = raw_pred
            converted = False

        st.write(f"Converted (expm1 applied?): {converted}")
        st.write("Predicted rent array:", pred_rent)

        # 10) Validate prediction
        val = pred_rent[0] if hasattr(pred_rent, "__len__") else float(pred_rent)
        if np.isnan(val) or val <= 0:
            st.warning("The predicted rent seems invalid (NaN or ≤ 0). See diagnostics above.")
            # show recommended fixes
            st.markdown("""
            **Possible causes / next steps**
            - Your model was trained with a different set of features (amenities you added are not part of training), causing unexpected predictions.  
            - The model output might not be in log1p scale; converting with `expm1` may be wrong. Try removing the `expm1` conversion.  
            - There may be mismatches between `encoder`, `scaler` and `model` (check they were saved/loaded from the same training pipeline).
            """)
            # Optional fallback: let user choose a minimum/fallback value
            fallback_min = st.number_input("Set a minimum fallback rent (optional). Leave 0 to skip:", value=0, min_value=0)
            if fallback_min > 0:
                st.success(f"Predicted Monthly Rent (fallback applied): ₹{int(max(fallback_min, 0)):,}")
            else:
                st.info("No fallback applied. Please check diagnostics above and fix training/pipeline.")
        else:
            st.success(f"**Predicted Monthly Rent:** ₹{round(val):,}")
            if selected_amenities:
                st.info(f"Amenities selected: {', '.join(selected_amenities)}")

    except Exception as e:
        st.error(f"Prediction failed with exception: {e}")

