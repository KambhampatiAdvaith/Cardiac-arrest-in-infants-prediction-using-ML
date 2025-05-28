import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Page Configuration ---
st.set_page_config(
    page_title="Infant Cardiac Arrest Predictor (Conceptual Prototype)",
    page_icon="👶❤️",
    layout="wide",
)

# --- Load The Trained Model Pipeline (STILL THE ADULT PROXY MODEL) ---
MODEL_PATH = 'best_cardiac_prediction_pipeline.pkl'
try:
    model_pipeline = joblib.load(MODEL_PATH)
except FileNotFoundError:
    st.error(f"CRITICAL ERROR: Model file '{MODEL_PATH}' not found. Dashboard cannot operate.")
    st.stop()
except Exception as e:
    st.error(f"Critical Error loading model: {e}")
    st.stop()

# --- Feature Names (STILL THE 28 FEATURES FROM THE ADULT MODEL) ---
# This list MUST match the features the loaded model_pipeline expects.
expected_feature_names = [
    'SBP', 'DBP', 'HR', 'RR', 'BT', 'SpO2', 'Age', 'Gender', 'GCS',
    'Alcoholic', 'Smoke', 'FHCD', 'TriageScore',
    'Urea', 'Cl', 'Na', 'K', 'Ceratinine',
    'Urea_missing', 'Cl_missing', 'Na_missing', 'K_missing', 'Ceratinine_missing',
    'PulsePressure', 'MAP',
    'AgeGroup_Age_65-74', 'AgeGroup_Age_75-84', 'AgeGroup_Age_85+'
    # ENSURE THIS LIST IS 100% ACCURATE FROM YOUR TRAINING DATA (X_train_lab.columns)
]

# --- Application Title & VERY STRONG DISCLAIMER ---
st.title("👶❤️ Conceptual Prototype: Early Infant Cardiac Arrest Risk Indicator")
st.error( # Using st.error for maximum visibility of the disclaimer
    """
    **⚠️ EXTREMELY IMPORTANT DISCLAIMER & USER AWARENESS ⚠️**

    *   **THIS IS A CONCEPTUAL PROTOTYPE FOR ILLUSTRATIVE PURPOSES ONLY.**
    *   The underlying predictive model was **TRAINED EXCLUSIVELY ON ADULT CARDIAC PATIENT DATA.**
    """
)
st.markdown("---")

# --- Sidebar for "Infant" Inputs (Focus on Clinically Plausible Infant Vitals) ---
st.sidebar.header("Hypothetical Infant Data Input:")
st.sidebar.info("Enter values typical for an infant. These will be used by the underlying adult proxy model.")

input_data_raw = {} # Store raw user inputs for infant-like parameters

st.sidebar.subheader("Core Infant Information & Vitals")
infant_age_days = st.sidebar.number_input("Infant Age (days)", min_value=0, max_value=365, value=30, step=1, help="e.g., 0-28 days for neonate, up to 365 for infant.")
input_data_raw['infant_age_days'] = infant_age_days

input_data_raw['Gender_infant'] = st.sidebar.selectbox("Infant Gender (0=Female, 1=Male)", options=[0, 1], index=1)

# Infant vital sign ranges (these are illustrative)
input_data_raw['HR_infant'] = st.sidebar.slider("Infant Heart Rate (HR, bpm)", 70, 220, 140)
input_data_raw['RR_infant'] = st.sidebar.slider("Infant Respiratory Rate (RR, breaths/min)", 20, 80, 40)
input_data_raw['SBP_infant'] = st.sidebar.slider("Infant Systolic BP (SBP, mmHg)", 50, 120, 75)
input_data_raw['DBP_infant'] = st.sidebar.slider("Infant Diastolic BP (DBP, mmHg)", 30, 80, 45)
input_data_raw['SpO2_infant'] = st.sidebar.slider("Infant Oxygen Saturation (SpO2, %)", 70, 100, 96, help="Ensure this is post-ductal if neonate, or as appropriate.")
input_data_raw['BT_infant'] = st.sidebar.slider("Infant Body Temp (°C)", 35.5, 38.5, 37.0, step=0.1)

# --- Mapping "INFANT" UI INPUTS TO THE ADULT MODEL'S EXPECTED FEATURES ---
input_data_for_model = {} # This will hold the 28 features for the adult model

# 1. Directly Mapped Vitals (values from infant UI inputs, feature names from adult model)
input_data_for_model['HR'] = input_data_raw['HR_infant']
input_data_for_model['RR'] = input_data_raw['RR_infant']
input_data_for_model['SBP'] = input_data_raw['SBP_infant']
input_data_for_model['DBP'] = input_data_raw['DBP_infant']
input_data_for_model['SpO2'] = input_data_raw['SpO2_infant']
input_data_for_model['BT'] = input_data_raw['BT_infant'] # Assume units are consistent or handled in a real scenario
input_data_for_model['Gender'] = input_data_raw['Gender_infant']

# 2. Proxy Adult Model Features (Set to fixed/neutral values or minimally interactive if needed for demo)

#   'Age' & 'AgeGroup_...' for adult model:
#   For this prototype, we'll use a fixed "young adult" age for the proxy model,
#   as infant age in days doesn't directly map to the adult model's learned patterns.
fixed_proxy_adult_age = 25 # A younger adult age, less likely to have age-related comorbidities from model's perspective
input_data_for_model['Age'] = fixed_proxy_adult_age
# Based on this fixed age, determine the AgeGroup for the adult model
input_data_for_model['AgeGroup_Age_65-74'] = 1 if (fixed_proxy_adult_age >= 65 and fixed_proxy_adult_age < 75) else 0
input_data_for_model['AgeGroup_Age_75-84'] = 1 if (fixed_proxy_adult_age >= 75 and fixed_proxy_adult_age < 85) else 0
input_data_for_model['AgeGroup_Age_85+'] = 1 if (fixed_proxy_adult_age >= 85) else 0

#   Adult-specific lifestyle/history features - set to "low risk" defaults for the proxy model
input_data_for_model['Alcoholic'] = 0.0
input_data_for_model['Smoke'] = 0.0
input_data_for_model['FHCD'] = 0.0 # Assuming no family history for this proxy input



#   Adult clinical scores - set to "good" defaults for the proxy model
input_data_for_model['GCS'] = 15.0 # Max GCS
input_data_for_model['TriageScore'] = 1.0 # Assuming 1 is least severe for proxy model (adjust if scale is different)



#   Adult Lab Values & Missingness Indicators for the proxy model:
#   These are complex as they are not direct infant equivalents for acute risk in this format.
#   We will set them to "normal adult" values and "not missing" for this conceptual prototype.

default_adult_labs = {'Urea': 20.0, 'Cl': 100.0, 'Na': 140.0, 'K': 4.0, 'Ceratinine': 1.0}
for lab_col_name in ['Urea', 'Cl', 'Na', 'K', 'Ceratinine']:
    input_data_for_model[lab_col_name] = default_adult_labs[lab_col_name]
    input_data_for_model[f'{lab_col_name}_missing'] = 0 # 0 = value is present (not missing)

# 3. Engineered features (PulsePressure, MAP) - calculated from the SBP/DBP inputs
if 'SBP' in input_data_for_model and 'DBP' in input_data_for_model:
    input_data_for_model['PulsePressure'] = input_data_for_model['SBP'] - input_data_for_model['DBP']
    input_data_for_model['MAP'] = (input_data_for_model['SBP'] + 2 * input_data_for_model['DBP']) / 3
else: # Should not happen if SBP/DBP are correctly mapped
    st.error("Error: SBP or DBP not available for derived feature calculation.")
    st.stop()


# --- Create DataFrame for Prediction & Make Prediction ---
st.markdown("---") # Visual separator in main panel
if st.sidebar.button("Predict Infant Risk (Conceptual Prototype)"):
    # Defensive check: Ensure all expected features are present in input_data_for_model
    all_features_present = True
    for feature_name in expected_feature_names:
        if feature_name not in input_data_for_model:
            st.error(f"Programming Error in app.py: Feature '{feature_name}' is missing from 'input_data_for_model'.")
            all_features_present = False
            break
    
    if all_features_present:
        try:
            # Create DataFrame in the exact order the model expects
            input_df = pd.DataFrame([input_data_for_model], columns=expected_feature_names)

            st.subheader("Prediction Results (Based on Adult Proxy Model - FOR ILLUSTRATION ONLY)")
            prediction = model_pipeline.predict(input_df)
            probability = model_pipeline.predict_proba(input_df)

            outcome_from_adult_model = prediction[0]
            prob_class1_adult_model = probability[0][1]

            # Interpret adult model's Class 1 as "Higher Conceptual Risk" for the infant prototype
            if outcome_from_adult_model == 1:
                st.error("Conceptual Risk Category: Higher Risk")
            else:
                st.success("Conceptual Risk Category: Lower Risk")

            st.markdown(f"**Estimated Probability of High Risk (Class 1):** `{prob_class1_adult_model:.2%}`")


            st.write(f"Conceptual Risk Score (Derived from Adult Proxy Model): {prob_class1_adult_model:.4f}")
            st.progress(prob_class1_adult_model)

            with st.expander("View Mapped Input Data Sent to Proxy Model"):
                st.dataframe(input_df.T.rename(columns={0: 'Mapped Value'}))
            
            st.caption("Reminder: This prediction is based on a model trained on adult data and is for demonstration only. Not for clinical use.")

        except Exception as e:
            st.error(f"Error during conceptual prediction: {e}")
            st.info("This dashboard is a prototype. Please check input mappings and model compatibility.")
            if 'input_df' in locals():
                 st.write("Data sent to model (first 5 columns):", input_df.iloc[:,:5].head())
    else:
        st.error("Could not proceed with prediction due to missing features in the input construction.")




# --- To run this app ---
# 1. Save this code as app.py
# 2. Open your terminal/command prompt
# 3. Navigate to the directory where app.py and your .pkl file are saved
# 4. Run: streamlit run app.py


