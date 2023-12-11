import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import os
import altair as alt
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.filterwarnings('ignore') 

# Save the initial state of the app
initial_state = {}


st.set_page_config(page_title="MortaliSys", page_icon=":stethoscope:", layout="wide")



# Dashboard Title
st.title("	:stethoscope: MortaliSys")
st.markdown('<style>div.block-container{padding-top:1rem;}</style>', unsafe_allow_html=True)

# st.cache
# Load dataset
fl = st.file_uploader(":file_folder: Upload a file", type=(["csv","txt","xlsx","xls"]))
if fl is not None:
    filename = fl.name
    st.write(filename)
    df = pd.read_csv(filename, encoding="ISO-8859-1")
else:
    os.chdir(r"C:\Users\sussa\OneDrive\Desktop\VA\MortaliSys\Dashboard")
    df = pd.read_csv("clinical_data.csv", encoding = "ISO-8859-1")


# Make HYPERTENSION categorical
def categorize_htn(htn_value):
    if htn_value == 0:
        return 'No'
    else:
        return 'Yes'

# Apply the function to create a new 'htn_category' column
htn_levels = ['No', 'Yes']
df['htn_category'] = df['preop_htn'].apply(categorize_htn).astype(pd.CategoricalDtype(categories=htn_levels, ordered=True))

# Make MORTALITY categorical
def categorize_death(death_value):
    if death_value == 0:
        return 'Alive'
    else:
        return 'Dead'

# Apply the function to create a new 'htn_category' column
death_levels = ['Alive', 'Dead']
df['death_category'] = df['death_inhosp'].apply(categorize_death).astype(pd.CategoricalDtype(categories=death_levels, ordered=True))

# Make DIABETES categorical
def categorize_dm(dm_value):
    if dm_value == 0:
        return 'No'
    else:
        return 'Yes'

# Apply the function to create a new 'htn_category' column
dm_levels = ['No', 'Yes']
df['dm_category'] = df['preop_dm'].apply(categorize_dm).astype(pd.CategoricalDtype(categories=dm_levels, ordered=True))

# Make AGE categorical
def categorize_age(age_value):
    if age_value < '18':
        return 'Below 18'
    elif '18' <= age_value < '30':
        return '18 - 29'
    elif '30' <= age_value < '46':
        return '30 - 45'
    else:
        return 'Above 45'

# Apply the function to create a new 'age_category' column
age_levels = ['Below 18', '18 - 29', '30 - 45', 'Above 45']
df['age_category'] = df['age'].apply(categorize_age).astype(pd.CategoricalDtype(categories=age_levels, ordered=True))

# Make BMI categorical
def categorize_bmi(bmi_value):
    if bmi_value < 18.5:
        return 'Underweight'
    elif 18.5 <= bmi_value < 24.9:
        return 'Normal'
    elif 25 <= bmi_value < 29.9:
        return 'Overweight'
    else:
        return 'Obese'

# Apply the function to create a new 'bmi_category' column
bmi_levels = ['Underweight', 'Normal', 'Overweight', 'Obese']
df['bmi_category'] = df['bmi'].apply(categorize_bmi).astype(pd.CategoricalDtype(categories=bmi_levels, ordered=True))

# Make hospital stay categorical
def categorize_stay(hospital_stay):
    if hospital_stay < 1:
        return 'Less than a day'
    elif 1 <= hospital_stay < 7:
        return 'Under a week'
    elif 7 <= hospital_stay < 30:
        return 'Under a month'
    else:
        return 'More than one month'

# Apply the function to create a new 'bmi_category' column
icu_days_levels = ['Less than a day', 'Under a week', 'Under a month', 'More than one month']
df['icu_category'] = df['icu_days'].apply(categorize_stay).astype(pd.CategoricalDtype(categories=icu_days_levels, ordered=True))


# DASHBOARD TABS
tab1, tab2 = st.tabs(["📊PATIENT DATA EXPLORATION", "🤖 PATIENT RISK ASSESSMENT"])

# Display dashboard overview
with st.sidebar.expander("Dashboard Overview"):
    st.subheader("Dataset Overview")
    st.write(f"Total Patients: {len(df)}")

    # Display total male and female patients
    total_male = df[df['sex'] == 'M'].shape[0]
    total_female = df[df['sex'] == 'F'].shape[0]
    st.write(f"Total Male Patients: {total_male}")
    st.write(f"Total Female Patients: {total_female}")

    # Display in-hospital mortality statistics
    total_deaths = df['death_inhosp'].sum()
    total_survivors = len(df) - total_deaths

    st.subheader("In-Hospital Mortality")
    st.write(f"Total Deaths: {total_deaths}")
    st.write(f"Total Survivors: {total_survivors}")

    # Calculate and display the percentage of in-hospital mortality
    mortality_percentage = (total_deaths / len(df)) * 100
    st.write(f"Percentage of In-Hospital Mortality: {mortality_percentage:.2f}%")

    # Calculate and display the minimum and maximum ICU days
    min_icu_days = df['icu_days'].min()
    max_icu_days = df['icu_days'].max()

    st.subheader("ICU Days")
    st.write(f"Minimum ICU Days: {min_icu_days}")
    st.write(f"Maximum ICU Days: {max_icu_days}")

    # Calculate and display the percentage of patients with the minimum and maximum ICU days
    min_icu_percentage = (df[df['icu_days'] == min_icu_days].shape[0] / len(df)) * 100
    max_icu_percentage = (df[df['icu_days'] == max_icu_days].shape[0] / len(df)) * 100

    st.write(f"Percentage of Patients with Minimum ICU Days ({min_icu_days}): {min_icu_percentage:.2f}%")
    st.write(f"Percentage of Patients with Maximum ICU Days ({max_icu_days}): {max_icu_percentage:.2f}%")

    st.subheader("Patient Demographics")

    # Replace non-numeric values in the 'age' column with 0
    df['age'] = pd.to_numeric(df['age'], errors='coerce').fillna(0)

    # Calculate and display the average age
    avg_age = df['age'].mean()
    st.write(f"Average Age: {avg_age:.2f} years")

    # Calculate and display the percentage of patients with hypertension and diabetes
    hypertensive_percentage = (df['preop_htn'].sum() / len(df)) * 100
    diabetic_percentage = (df['preop_dm'].sum() / len(df)) * 100
    st.write(f"Percentage of Patients with Hypertension: {hypertensive_percentage:.2f}%")
    st.write(f"Percentage of Patients with Diabetes: {diabetic_percentage:.2f}%")

    st.subheader("Surgical Details")
    st.write(f"Total Departments: {df['department'].nunique()}")
    st.write(f"Total Operation Types: {df['optype'].nunique()}")
    st.write(f"Total Approaches: {df['approach'].nunique()}")

# Side Filter pane
st.sidebar.header("Choose your filter: ")

# Select outcome
# outcomes = {'Inhospital Mortality': 'death_inhosp', 'Length of Hospital Stay': 'icu_days'}
outcomes = {'Inhospital Mortality': 'death_category', 'Length of Hospital Stay': 'icu_category'}
selected_outcome = st.sidebar.selectbox('Outcomes to explore', list(outcomes.keys()))

# Check if a valid outcome is selected
if selected_outcome in outcomes:
    metric_to_show = outcomes[selected_outcome]
else:
    st.warning("Please select a valid outcome.")

# Select surgery detail
details = {'Department': 'department', 'Operation Type': 'optype','Approach': 'approach', 'Anesthesia Type': 'ane_type'}
selected_detail = st.sidebar.selectbox('Surgery Details to explore', list(details.keys()))

# Check if a valid outcome is selected
if selected_detail in details:
    detail_to_show = details[selected_detail]
else:
    st.warning("Please select a valid detail.")


# Select demographics
# demographics = {'Age': 'age', 'BMI': 'bmi', 'Weight': 'weight','Height': 'height'}
demographics = {'Age': 'age_category', 'BMI': 'bmi_category', 'Gender': 'sex'}
selected_demographic = st.sidebar.selectbox('Patient Demographics to explore', list(demographics.keys()))

# Check if a valid outcome is selected
if selected_demographic in demographics:
    demographic_to_show = demographics[selected_demographic]
else:
    st.warning("Please select a valid detail.")


# Select medical history
history = {'Hypertension': 'htn_category', 'Diabetes': 'dm_category','Pulmonary': 'preop_pft'}
selected_history = st.sidebar.selectbox('Medical History', list(history.keys()))

# Check if a valid outcome is selected
if selected_history in history:
    history_to_show = history[selected_history]
else:
    st.warning("Please select a valid detail.")


# Reset button in the sidebar
reset_button = st.sidebar.button("Reset Dashboard")


if reset_button:
       # Trigger a complete rerun of the app
    st.rerun()



with tab1:
  # A. CLUSTERED COLUMN CHART: Sum of ICU Days and Death In-Hospital by Surgery Details and Operation Name
     
    grouped_data = df.groupby([detail_to_show, 'opname', demographic_to_show, 'icu_category']).agg({
        'icu_days': 'sum',
        'death_inhosp': 'sum'
    }).reset_index()

    # Create a clustered column chart
    fig_clustered_column = px.bar(
        grouped_data,
        x='death_inhosp',
        y=detail_to_show,
        color='opname',
        facet_row=demographic_to_show,
        facet_col='icu_category',
        hover_data=[detail_to_show, 'opname', 'icu_category', 'death_inhosp', demographic_to_show],
        title='Clustered Column Chart: Length of Hospital Stay and Death In-Hospital by Surgery Details and Patient Demographics',
        labels={'death_inhosp': 'Number of Dead patients', detail_to_show: f'{selected_detail}', 'opname': 'Operation Name','icu_category':'Duration', demographic_to_show: f'{selected_demographic}', history_to_show:f'{selected_history}'},
        orientation='h'
    )

    # Show the chart
    st.plotly_chart(fig_clustered_column, use_container_width=True)


    # B. CLUSTERED COLUMN CHART: Sum of ICU Days and Death In-Hospital by Patient Demographics and Medical History
    # Group by patient demographic, history_to_show and calculate the sum of icu_days and death_inhosp
    grouped_data = df.groupby([demographic_to_show, history_to_show, 'icu_category']).agg({
        'icu_days': 'sum',
        'death_inhosp':'sum',
    }).reset_index()

    # Create a clustered column chart
    fig_clustered_histogram = px.histogram(
        grouped_data,
        x='death_inhosp',
        y=demographic_to_show,
        facet_col='icu_category',
        # facet_col='age_category',
        color=history_to_show,
        hover_data=[demographic_to_show, history_to_show, 'icu_days', 'death_inhosp'],
        title='Histogram: Death In-Hospital and Length of Hospital Stay by Patient Demographics and Medical History',
        labels={'death_inhosp': 'number of dead patients', demographic_to_show: f'{selected_demographic}', history_to_show:f'{selected_history}', 'icu_category':'Duration'},
        orientation='h'
    )

    # Show the chart
    st.plotly_chart(fig_clustered_histogram, use_container_width=True)


# 3. SCATTER PLOT: Correlation of age and bmi in determining Inhospital Mortality and Length of Hospital Stay
     
    fig = px.scatter(df, 
                     y='age', 
                     x='bmi', 
                     color=metric_to_show,
                     marginal_x="histogram", 
                     marginal_y="rug",
                     facet_col="sex",
                     title='Scatter Plot: Correlation of Age and Body Mass Index in determining Inhospital Mortality and Length of Hospital Stay',
                     hover_data=['age_category', history_to_show, metric_to_show, 'bmi_category', 'ane_type'],
                     labels={'age': 'Age', metric_to_show: f'{selected_outcome}', history_to_show: f'{selected_history}' ,'bmi':'Body Mass Index', 'bmi_category':'Body Mass Index','age_category':'Age range' },
                     )
    # Show the chart
    st.plotly_chart(fig, use_container_width=True) 





