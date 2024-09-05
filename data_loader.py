import pandas as pd
import streamlit as st


@st.cache_data
def load_data(file):
    # Define the columns of interest
    columns_to_load = ['motherAge', 'Height', 'BMI', 'wtGain', 'Week', 'Cigarette',
          'noPrenatal', 'preDiab', 'gestDiab', 'preHype', 'gestHype', 'noInfec',
          'Boy', 'infTreatment', 'Smoker', 'Parity', 'Black', 'Weight', 'BirthWeightCategory']
    
    # Load and return the data
    return pd.read_csv(file, usecols=columns_to_load)

# Mapping of dataset variable names to user-friendly names
feature_names_mapping = {
    'motherAge': 'Mother Age',
    'Height': 'Height',
    'BMI': 'Body Mass Index (BMI)',
    'wtGain': 'Weight Gain',
    'Week': 'Week of Pregnancy',
    'Cigarette': 'Cigarettes Per Day',
    'noPrenatal': 'No Prenatal Care',
    'preDiab': 'Pre-existing Diabetes',
    'gestDiab': 'Gestational Diabetes',
    'preHype': 'Pre-existing Hypertension',
    'gestHype': 'Gestational Hypertension',
    'noInfec': 'No Pregnancy-related Infection',
    'Boy': 'Gender (Boy)',
    'infTreatment': 'Infection Treatment',
    'Smoker': 'Smoker',
    'Parity': 'Parity (Previous Births)',
    'Black': 'Ethnicity (Black)'

}

# Define the categorical keys
categorical_keys = ['Smoker', 'gestDiab', 'preDiab', 'preHype', 'gestHype', 'noPrenatal', 'Parity', 'infTreatment', 'noInfec', 'Black', 'Boy']
# Define the continuous keys
continuous_keys = ['motherAge', 'BMI', 'Height','wtGain', 'Week', 'Cigarette']