import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns  
import matplotlib.pyplot as plt 
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, roc_curve, roc_auc_score
import pandas as pd
import os
import altair as alt
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore') 

st.set_page_config(page_title="MortaliSys", page_icon=":stethoscope:", layout="wide")

uploaded_file = st.file_uploader(":file_folder: Upload a file", type=(["csv"]))
if uploaded_file is not None:
    datafile = uploaded_file
else:
    datafile = "data/clinical_data_cleaned.csv"

@st.cache_data
def load_data(file):
    columns_to_load = ['weight', 'height', 'bmi', 'preop_htn', 'preop_dm', 'age', 'age_category', 'age_category', 'sex', 'htn_category', 'dm_category', 'preop_pft', 'bmi_category', 'ane_type', 'department', 'optype', 'approach', 'opname', 'icu_days', 'death_inhosp', 'icu_category', 'death_category']

    # Load the data
    return pd.read_csv(file, usecols=columns_to_load, encoding = "ISO-8859-1")

df = load_data(datafile)

# Dashboard Title
st.title("	:stethoscope: MortaliSys")
st.markdown('<style>div.block-container{padding-top:1rem;}</style>', unsafe_allow_html=True)


# DASHBOARD TABS
tab1, tab2 = st.tabs(["📊 PATIENT DATA EXPLORATION", "🤖 PATIENT RISK ASSESSMENT"])

# Display dashboard overview
with st.sidebar.expander("Statistics Overview"):
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
with st.sidebar.expander("Exploration Filter"):
    # Select surgery detail
    details = {'Department': 'department', 'Operation Type': 'optype','Approach': 'approach', 'Anesthesia Type': 'ane_type'}
    selected_detail = st.sidebar.selectbox('Surgery Details to explore', list(details.keys()))

    # Check if a valid outcome is selected
    if selected_detail in details:
        detail_to_show = details[selected_detail]
    else:
        st.warning("Please select a valid detail.")


    # Select demographics
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

    # Select outcome
    outcomes = {'Inhospital Mortality': 'death_category', 'Length of Hospital Stay': 'icu_category'}
    selected_outcome = st.sidebar.selectbox('Outcomes to explore', list(outcomes.keys()))

    # Check if a valid outcome is selected
    if selected_outcome in outcomes:
        metric_to_show = outcomes[selected_outcome]
    else:
        st.warning("Please select a valid outcome.")


    # Reset button in the sidebar
    reset_button = st.sidebar.button("Reset Dashboard")

# Patient Risk Assesment
st.sidebar.header("Patient Risk Assessment")
with st.sidebar.expander("Patient Information for Prediction"):
    # Fields for user input
    with st.form("prediction_form"):
        age_input = st.number_input("Age", min_value=1, max_value=100, step=1, value=30)
        bmi_input = st.number_input("BMI", min_value=10.0, max_value=50.0, step=0.1, value=25.0)
        sex_input = st.radio("Gender", ['M', 'F'])  # Modified this line
        department_input = st.selectbox("Department", df['department'].unique())
        optype_input = st.selectbox("Operation Type", df['optype'].unique())
        approach_input = st.selectbox("Approach", df['approach'].unique())
        ane_type_input = st.selectbox("Anesthesia Type", df['ane_type'].unique())
        htn_input = st.radio("Hypertension", ['No', 'Yes'])
        dm_input = st.radio("Diabetes", ['No', 'Yes'])
        pft_input = st.radio("Pulmonary", ['No', 'Yes'])
        
        predict_button = st.form_submit_button("Predict")

@st.cache_data
def get_grouped_data(features):
    return df.groupby(features).agg({
        'icu_days': 'sum',
        'death_inhosp': 'sum'
    }).reset_index()

# CLUSTERED COLUMN CHART
@st.cache_data
def get_clustered_column_plot(detail_to_show, demographic_to_show, history_to_show, selected_demographic):
    grouped_data = get_grouped_data([detail_to_show, 'opname', demographic_to_show, 'icu_category'])

    return px.bar(
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

# HISTOGRAM
@st.cache_data
def get_clustered_histogram(demographic_to_show, history_to_show, selected_demographic):
    grouped_data = get_grouped_data([demographic_to_show, history_to_show, 'icu_category'])

    return px.histogram(
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
# SCATTERPLOT
@st.cache_data
def get_scatterplot(metric_to_show, history_to_show, selected_outcome, selected_history):
    return px.scatter(
        df, 
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

# HEATMAP
@st.cache_data
def get_heatmap(features_for_correlation, corr_matrix, selected_features, color_scale):
   
    # Create an interactive heatmap using Plotly with the selected color scale
    return go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=selected_features,
        y=selected_features,
        colorscale=color_scale,
        colorbar=dict(title='Correlation'),
    ))

# HEATMAP UPDATE
@st.cache_data
def get_heatmap_update(fig_heatmap, corr_matrix):
   
    # Create an interactive heatmap using Plotly with the selected color scale
    return fig_heatmap.update_traces(
        hovertemplate='%{y} vs %{x}: %{z:.2f}<br>' +
        'Interpretation: %{customdata}',
        customdata=[
            [
                'Strong correlation' if abs(val) >= 0.7 else
                'Moderate correlation' if 0.5 <= abs(val) < 0.7 else
                'Weak correlation' if 0.3 <= abs(val) < 0.5 else
                'No correlation'
                for val in row
            ]
            for row in corr_matrix.values
        ],
    )

# HEATMAP LAYOUT
@st.cache_data
def get_heatmap_layout(fig_heatmap):
   
    # Create an interactive heatmap using Plotly with the selected color scale
    return fig_heatmap.update_layout(
        title='Interactive Feature Correlation Analysis',
        height=600, width=800
    )



with tab1:
    # 1. CLUSTERED COLUMN CHART: Sum of ICU Days and Death In-Hospital by Surgery Details and Operation Name

    # Create a clustered column chart
    fig_clustered_column = get_clustered_column_plot(detail_to_show, demographic_to_show, history_to_show, selected_demographic)

    # Show the chart
    st.plotly_chart(fig_clustered_column, use_container_width=True)


    # 2. CLUSTERED HISTOGRAM: Sum of ICU Days and Death In-Hospital by Patient Demographics and Medical History

    # Create a clustered column chart
    fig_clustered_histogram = get_clustered_histogram(demographic_to_show, history_to_show, selected_demographic)

    # Show the chart
    st.plotly_chart(fig_clustered_histogram, use_container_width=True)


    # 3. SCATTER PLOT: Correlation of age and bmi in determining Inhospital Mortality and Length of Hospital Stay
     
    fig_scatterplot = get_scatterplot(metric_to_show, history_to_show, selected_outcome, selected_history)

    # Show the chart
    st.plotly_chart(fig_scatterplot, use_container_width=True) 

    # 4. HEATMAP

    # Display feature correlation analysis using an interactive heatmap
    st.markdown("**Feature Correlation Analysis**")
    # Specify the features for correlation analysis with aliases
    features_for_correlation = {'Duration' : 'icu_days', 'Age' : 'age', 'Weight' : 'weight',  'Height': 'height', 'Hypertension' : 'preop_htn', 'BMI' : 'bmi', 'Diabetes' : 'preop_dm', 'Mortality' : 'death_inhosp'}
    
    # Create a subset of the DataFrame with the specified features
    subset_df = df[features_for_correlation.values()]
    
    # Feature selection dropdown
    selected_features = st.multiselect('Select Features for Correlation Analysis', list(features_for_correlation.keys()), default=list(features_for_correlation.keys()))


    # Color scale adjustment
    color_scale = st.selectbox('Select Color Scale', options=['Viridis', 'Plasma', 'Inferno', 'Magma', 'Cividis'])


    # Exclude non-numeric columns before calculating the correlation matrix
    numeric_columns = subset_df.select_dtypes(include=['number']).columns
    corr_matrix = subset_df[numeric_columns].corr()


    fig_heatmap  = get_heatmap(features_for_correlation, corr_matrix, selected_features, color_scale)

    # Customize the hover information for each tile dynamically
    get_heatmap_update(fig_heatmap, corr_matrix)

    get_heatmap_layout(fig_heatmap)

    # Show the interactive heatmap
    st.plotly_chart(fig_heatmap)


# MACHINE LEARNING SECTION STARTS HERE:

# Separate features (X) and target variable (y)
X = df[['age', 'bmi', 'sex', 'department', 'optype', 'approach', 'ane_type', 'preop_htn', 'preop_dm', 'preop_pft']]
y_mortality = df['death_inhosp']
y_icu = df['icu_category']

# Define categorical and numeric features
categorical_features = ['sex', 'department', 'optype', 'approach', 'ane_type', 'preop_pft']
numeric_features = ['age', 'bmi', 'preop_htn', 'preop_dm']

# Create transformers for one-hot encoding and scaling
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ]
)

# Resample the data for mortality prediction
ros_mortality = RandomOverSampler(random_state=42)
X_resampled_mortality, y_resampled_mortality = ros_mortality.fit_resample(X, y_mortality)

# Resample the data for ICU stay prediction
ros_icu = RandomOverSampler(random_state=42)
X_resampled_icu, y_resampled_icu = ros_icu.fit_resample(X, y_icu)

# Split the data into training and testing sets
X_train_mortality, X_test_mortality, y_train_mortality, y_test_mortality = train_test_split(X_resampled_mortality, y_resampled_mortality, test_size=0.2, random_state=42)
X_train_icu, X_test_icu, y_train_icu, y_test_icu = train_test_split(X_resampled_icu, y_resampled_icu, test_size=0.2, random_state=42)

# Train the models
model_mortality = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', SVC(kernel='rbf', random_state=42, class_weight='balanced'))
])

model_icu = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

model_mortality.fit(X_train_mortality, y_train_mortality)
model_icu.fit(X_train_icu, y_train_icu)

# Evaluate the models
y_pred_mortality = model_mortality.predict(X_test_mortality)
y_pred_icu = model_icu.predict(X_test_icu)

accuracy_mortality = accuracy_score(y_test_mortality, y_pred_mortality)
accuracy_icu = accuracy_score(y_test_icu, y_pred_icu)

# Calculate additional metrics
f1_length_of_stay = f1_score(y_test_icu, y_pred_icu, average='weighted')
recall_length_of_stay = recall_score(y_test_icu, y_pred_icu, average='weighted')
precision_length_of_stay = precision_score(y_test_icu, y_pred_icu, average='weighted')


print(f"Accuracy for Mortality Prediction: {accuracy_mortality}")
print(f"Accuracy for ICU Stay Prediction: {accuracy_icu}")


with tab2:
    st.title("MortaliSys - Patient Outcome Prediction")


    if predict_button:
        # Combine user input into a DataFrame
        user_input = pd.DataFrame({
            'age': [age_input],
            'bmi': [bmi_input],
            'sex': [sex_input],
            'department': [department_input],
            'optype': [optype_input],
            'approach': [approach_input],
            'ane_type': [ane_type_input],
            'preop_htn': [1 if htn_input == 'Yes' else 0],
            'preop_dm': [1 if dm_input == 'Yes' else 0],
            'preop_pft': [1 if pft_input == 'Yes' else 0],
        })

        # Handle NaN values in the user input DataFrame
        user_input = user_input.fillna(0)  # Replace NaN values with a specific value (you can choose a different value)

        # Ensure data consistency for categorical features
        user_input['sex'] = user_input['sex'].astype(str)  # Ensure 'sex' is of type string
        user_input['department'] = user_input['department'].astype(str)
        user_input['optype'] = user_input['optype'].astype(str)
        user_input['ane_type'] = user_input['ane_type'].astype(str)
        user_input['preop_pft'] = user_input['preop_pft'].astype(str)

        # Handle unknown categories by replacing them with the most frequent category from the training data
        for col in categorical_features:
            # Get the most frequent category from the training data
            most_frequent_category = X[col].mode()[0]

            # Replace unknown categories with the most frequent category
            user_input[col] = user_input[col].apply(lambda x: x if x in X[col].unique() else most_frequent_category)

        # Predictions
        mortality_prediction = model_mortality.predict(user_input)[0]
        icu_prediction = model_icu.predict(user_input)[0]

        # Display predictions
        st.subheader("Outcome Predictions:")
        result_message = f"In-Hospital Mortality Prediction: {'Dead' if mortality_prediction == 1 else 'Alive'}\n\nLength of Hospital Stay Prediction: {icu_prediction}"
        st.info(result_message)

    # Evaluation metrics section
    evaluate_button = st.sidebar.button("Evaluate Model")


    if evaluate_button:
        # Evaluate the models
        y_scores_mortality = model_mortality.decision_function(X_test_mortality)
        y_proba_mortality = 1 / (1 + np.exp(-y_scores_mortality))

        y_scores_mortality = model_mortality.decision_function(X_test_mortality)
        y_proba_mortality = 1 / (1 + np.exp(-y_scores_mortality))
        fpr_mortality, tpr_mortality, _ = roc_curve(y_test_mortality, y_proba_mortality)


        # Display ROC curve for mortality prediction
        fig_mortality, ax_mortality = plt.subplots()
        y_scores_mortality = model_mortality.decision_function(X_test_mortality)
        fpr_mortality, tpr_mortality, _ = roc_curve(y_test_mortality, y_scores_mortality)
        auc_score_mortality = roc_auc_score(y_test_mortality, y_scores_mortality)

        plt.plot(fpr_mortality, tpr_mortality, label=f'AUC = {auc_score_mortality:.2f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve - Mortality Prediction')
        plt.legend(loc='lower right')
        
        # Display the ROC curve for mortality using st.pyplot()
        st.pyplot(fig_mortality)

        # Display classification report for mortality prediction
        st.subheader("Classification Report - Mortality Prediction:")

        # Create a DataFrame from the classification report for better formatting
        classification_df_mortality = pd.DataFrame(classification_report(y_test_mortality, y_pred_mortality, output_dict=True)).T

        # Display precision, recall, and F1-score in a tabular format
        st.dataframe(classification_df_mortality[['precision', 'recall', 'f1-score']])
        
        # Display support in a separate section
        st.subheader("Support - Mortality Prediction:")
        st.text("Support refers to the number of actual occurrences of the class in the specified dataset.")
        st.text(f"Support for class 0 (Alive): {classification_df_mortality.loc['0', 'support']}")
        st.text(f"Support for class 1 (Dead): {classification_df_mortality.loc['1', 'support']}")

        # Provide a user-friendly interpretation
        st.subheader("Interpretation - Mortality Prediction:")
        st.markdown("""
            - **Precision (Positive Predictive Value):** Indicates the accuracy of the positive predictions. 
            - **Recall (Sensitivity, True Positive Rate):** Measures the ability of the model to capture all positive instances.
            - **F1-score:** Balances precision and recall, providing a single metric for model evaluation.
        """)

        st.markdown("""
            In simple terms, precision tells us how often the model correctly predicted death when it predicted death. 
            Recall indicates how often the model correctly predicted death out of all actual deaths.
            The F1-score is a balanced measure that considers both precision and recall.
        """)

        
        # Length of Hospital Stay metrics
        st.subheader("Evaluation Metrics - Length of Hospital Stay:")

        # Display classification report for mortality prediction
        st.subheader("Classification Report - Length of Hospital Stay Prediction:")

        # Create a DataFrame from the classification report for better formatting
        classification_df_icu = pd.DataFrame(classification_report(y_test_icu, y_pred_icu, output_dict=True)).T

        # Display precision, recall, and F1-score in a tabular format
        st.dataframe(classification_df_icu[['precision', 'recall', 'f1-score']])

        # # Display support in a separate section
        # st.subheader("Support - Mortality Prediction:")
        # st.text("Support refers to the number of actual occurrences of the class in the specified dataset.")
        # st.text(f"Support for class Less than a day: {classification_df_icu.loc['Less than a day', 'support']}")
        # st.text(f"Suppor for More than a month: {classification_df_icu.loc['More than a month', 'support']}")
        # st.text(f"Suppor for Under a month: {classification_df_icu.loc['Under a month', 'support']}")
        # st.text(f"Suppor for Under a week: {classification_df_icu.loc['Under a week', 'support']}")

        # Provide a user-friendly interpretation for Length of Hospital Stay
        st.subheader("Interpretation - Length of Hospital Stay:")
        st.markdown("""
            - **F1-score:** A balanced measure that considers both precision and recall for Length of Hospital Stay prediction.
            - **Recall:** Measures the ability of the model to capture all instances of prolonged hospital stay.
            - **Precision:** Indicates the accuracy of the positive predictions for prolonged hospital stay.
        """)

        st.markdown("""
            In simple terms, the F1-score provides a balanced assessment of the model's ability to predict prolonged hospital stay.
            Recall tells us how often the model correctly predicted prolonged hospital stay out of all actual cases.
            Precision indicates how accurately the model predicted prolonged hospital stay when it made positive predictions.
        """)
