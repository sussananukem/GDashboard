import pandas as pd

# Read the data
df = pd.read_csv("data/clinical_data.csv", encoding = "ISO-8859-1")

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


# Export the cleaned data
df.to_csv('data/clinical_data_cleaned.csv', index=False)