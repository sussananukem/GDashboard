import streamlit as st
import plotly.express as px
import plotly.graph_objs as go
from data_loader import categorical_keys, continuous_keys


# VISUALIZATION FUNCTION A. Feature vs Birth Weight
def plot_feature_vs_birthweight(df, feature_key, feature_label):
    # Check if the feature is continuous or categorical
    if feature_key in continuous_keys:
        # Continuous feature plotting
        fig = px.box(df, x='BirthWeightCategory', y=feature_key, color='BirthWeightCategory',
                     title=f"Impact of {feature_label} on Birth Weight Category",
                     labels={'BirthWeightCategory':'Birth Weight Category', feature_key:feature_label}, color_discrete_sequence=px.colors.qualitative.Plotly)
    elif feature_key in categorical_keys:
        # Categorical feature plotting
        proportions = df.groupby([feature_key, 'BirthWeightCategory']).size() / df.groupby(feature_key).size()
        proportions = proportions.reset_index(name='Proportion')
        fig = px.bar(proportions, x=feature_key, y='Proportion', color='BirthWeightCategory',
                     title=f"Proportion of Birth Weight Category within {feature_label}",
                     labels={'Proportion':'Proportion', feature_key:feature_label, 'BirthWeightCategory':'Birth Weight Category'}, color_discrete_sequence=px.colors.qualitative.Plotly)
    else:
        st.error("Feature not recognized")
        return None  # Return None if feature not recognized
    
    fig.update_layout(transition_duration=500)
    return fig  # Return the figure instead of plotting it here




# VISUALIZATION FUNCTION B. Faceted Scatter Plot
def create_faceted_scatter_plot(df, categorical_column, continuous_column, continuous_feature_name, categorical_feature_name):
    fig = px.scatter(df, 
                     x=continuous_column, 
                     y='Weight', 
                     color='BirthWeightCategory', 
                     facet_col=categorical_column, 
                     title=f"Impact of {continuous_feature_name} on Baby Weight across {categorical_feature_name}",
                     labels={
                         continuous_column: continuous_feature_name,
                         'Weight': 'Baby Weight',
                         'BirthWeightCategory': 'Birth Weight Category',
                         categorical_column: categorical_feature_name
                     },
                     color_continuous_scale=px.colors.sequential.Viridis)
    fig.update_layout(height=600, width=800)  # Adjust size as needed
    return fig


# VISUALIZATION FUNCTION C. Correlation Heatmap

def correlation_interpretation(value):
    """Returns a textual interpretation of a correlation value."""
    if value > 0.7:
        return 'Strong positive correlation'
    elif value > 0.3:
        return 'Moderate positive correlation'
    elif value > 0:
        return 'Weak positive correlation'
    elif value == 0:
        return 'No correlation'
    elif value > -0.3:
        return 'Weak negative correlation'
    elif value > -0.7:
        return 'Moderate negative correlation'
    else:
        return 'Strong negative correlation'

def plot_correlation_heatmap(df, feature_names_mapping):
    
    # Create a copy of the DataFrame
    df_copy = df.copy()

    # Encoding for binary categorical variables
    binary_columns = [col for col in df_copy.columns if df_copy[col].dropna().isin(['No', 'Yes']).all()]
    for col in binary_columns:
        df_copy[col] = df_copy[col].map({'Yes': 1, 'No': 0})

    # Drop the BirthWeightCategory column from the copy
    df_copy = df_copy.drop(columns=['BirthWeightCategory'])

    # Calculate the correlation matrix for the modified DataFrame
    correlation_matrix = df_copy.corr()

    # Map raw variable names to user-friendly names for x and y axes
    friendly_names_x = [feature_names_mapping.get(col, col) for col in correlation_matrix.columns]
    friendly_names_y = [feature_names_mapping.get(col, col) for col in correlation_matrix.index]

    # Generate the heatmap
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=friendly_names_x,
        y=friendly_names_y,
        colorscale="RdBu_r",
        colorbar=dict(title='Correlation'),
    ))

    # Add custom hover data for correlation interpretation
    customdata = correlation_matrix.applymap(correlation_interpretation).values

    # Update the traces with custom hover data for interpretation
    fig_heatmap.update_traces(
        customdata=customdata,
        hovertemplate='%{y} vs %{x}: %{z:.2f}<br><extra>%{customdata}</extra>'
    )

    # Update layout for a better presentation
    fig_heatmap.update_layout(
        title='Interactive Feature Correlation Analysis',
        height=600,
        width=800,
        xaxis={'side': 'bottom'}
    )

    # Return the figure
    return fig_heatmap



