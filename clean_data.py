import pandas as pd

# Read the data
df = pd.read_csv("data/birthwt_2019.csv", encoding = "ISO-8859-1")

# Make BIRTH WEIGHT categorical
def categorize_weight(row):
    if row['Weight'] < 2500:
        return 'Low'
    elif row['Weight'] <= 4000:
        return 'Normal'
    else:
        return 'High'

# Apply the function to create a new categorical variable
df['BirthWeightCategory'] = df.apply(categorize_weight, axis=1)

# Detect binary variables and transform them
binary_columns = [col for col in df if df[col].dropna().isin([0, 1]).all()]
for col in binary_columns:
    df[col] = df[col].map({1: 'Yes', 0: 'No'})


# Export the cleaned data
df.to_csv('data/birthwt_cleaned.csv', index=False)