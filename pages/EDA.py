# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 15:57:46 2023

@author: raywh
"""

import dash
from dash import Dash, html, dcc, callback, Output, Input

import pandas as pd

import plotly.express as px


dash.register_page(__name__)


#Data imports
filepath = 'Data/'

#Importing the binary diabetes data set
df = pd.read_csv(f'{filepath}diabetes_binary_health_indicators_BRFSS2015.csv')

# Load the 012 dataset
df_012 = pd.read_csv(f'{filepath}diabetes_012_health_indicators_BRFSS2015.csv')

data = df

#Identify Binary vs continuous features
#Binary Variables
binary_variables = ['HighBP', 'HighChol', 'CholCheck', 'Smoker',
       'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
       'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'DiffWalk', 'Sex']

#Continuous Variables
boxplot_variables = ['BMI', 'GenHlth',
       'MentHlth', 'PhysHlth', 'Age', 'Education',
       'Income']

violinplot_variables = []

variables = data.columns[1:]#binary_variables + boxplot_variables

continuous_variables = ['BMI', 'GenHlth',
       'MentHlth', 'PhysHlth', 'Age', 'Education',
       'Income']


layout = html.Div([
    html.H1("Binary Variable vs. Diabetes Status"),
    dcc.Dropdown(
        id='variable-dropdown',
        options=[{'label': col, 'value': col} for col in variables],
        value=variables[0] if len(variables) > 1 else None
    ),
    #html.Div(id='plot-container')#,
    dcc.Graph(id='violinplot')
])
# Define callback to update the countplot based on dropdown value
@callback(
    #dash.dependencies.Output('plot-container', 'children'),
    dash.dependencies.Output('violinplot', 'figure'),
    [dash.dependencies.Input('variable-dropdown', 'value')]
)
def update_plot(selected_variable):
    if selected_variable in binary_variables:
        fig = px.histogram(data, x=selected_variable, color='Diabetes_binary', barmode='group',
                        title=f'{selected_variable.capitalize()} Distribution by Diabetes Status',
                        labels={'diabetes_binary': 'Diabetes Status'},
                        width=800, height=800)
        return fig
    elif selected_variable in continuous_variables:
        fig = px.violin(data, y=selected_variable, x='Diabetes_binary', box=True, points=False,
                        title=f'{selected_variable.capitalize()} Distribution by Diabetes Status',
                        labels={'diabetes_binary': 'Diabetes Status'},
                        width=800, height=800)
        return fig