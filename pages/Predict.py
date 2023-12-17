# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 15:58:13 2023

@author: raywh
"""
import dash
#import dash_core_components as dcc
#import dash_html_components as html
from dash import Dash, html, dcc, callback, Output, Input
#from dash.dependencies import Input, Output

#import plotly.express as px
import pickle
import numpy as np


dash.register_page(__name__)

pickle_path = 'PickleJar/'

# Load the pickled logistic regression model
with open(f'{pickle_path}/scaler.pkl', 'rb') as scale_file:
    scaler = pickle.load(scale_file)

with open(f'{pickle_path}/logistic_reg.pkl', 'rb') as log_file:
    log_reg = pickle.load(log_file)

#with open(f'{pickle_path}/random_forest.pkl', 'rb') as rand_file:
 #   rand_for = pickle.load(rand_file)
    

models = {'Logistic Regression': log_reg}
#          'Random Forest': rand_for}
    
# Define the value/label pairs for the 'income' variable
age_options = [
    {'label': '18 to 24', 'value': 1},
    {'label': '25 to 29', 'value': 2},
    {'label': '30 to 34', 'value': 3},
    {'label': '35 to 39', 'value': 4},
    {'label': '40 to 44', 'value': 5},
    {'label': '45 to 49', 'value': 6},
    {'label': '50 to 54', 'value': 7},
    {'label': '55 to 59', 'value': 8},
    {'label': '60 to 64', 'value': 9},
    {'label': '65 to 69', 'value': 10},
    {'label': '70 to 74', 'value': 11},
    {'label': '75 to 79', 'value': 12},
    {'label': '80 or older', 'value': 13}
]
    
# Define the value/label pairs for the 'income' variable
income_options = [
    {'label': 'Less than $10,000', 'value': 1},
    {'label': '$10,000-$15,000', 'value': 2},
    {'label': '$15,000-$20,000', 'value': 3},
    {'label': '$20,000-$25,000', 'value': 4},
    {'label': '$25,000-$35,000', 'value': 5},
    {'label': '$35,000-$50,000', 'value': 6},
    {'label': '$50,000-$75,000', 'value': 7},
    {'label': 'More than $75,000', 'value': 8}
]

# Define the value/label pairs for the 'education' variable
education_options = [
    {'label': 'No School', 'value': 1},
    {'label': 'Elementary', 'value': 2},
    {'label': 'Some High School', 'value': 3},
    {'label': 'High School Graduate', 'value': 4},
    {'label': 'Some College, Associates, or Technical School', 'value': 5},
    {'label': '4 Year College Graduate', 'value': 6},
]

health_options = [
    {'label': 'Excellent', 'value': 1},
    {'label': 'Very Good', 'value': 2},
    {'label': 'Good', 'value': 3},
    {'label': 'Fair', 'value': 4},
    {'label': 'Poor', 'value': 5},
    {'label': 'Not Sure', 'value': 7},
]

# Define the layout of the app
layout = html.Div([
    html.H1("Diabetes Predictor"),
    html.Div([
        #####Check these values!!!
        html.Label('Sex:'),
        dcc.RadioItems(
            id='Sex',
            options=[
                {'label': 'Female', 'value': 1},
                {'label': 'Male', 'value': 0}
            ],
            value=0
        ),        
        html.Label('BMI:'),
        dcc.Slider(
            id='BMI',
            min=10,
            max=50,
            step=1,
            value=25,  # Set default value
            marks={i: str(i) for i in range(10, 51, 5)}  # Display marks at intervals of 10
        ),
        html.Label('Age:'),
        dcc.Dropdown(
            id='Age',
            options=age_options,
            value=6  # Set default value
        ),
        html.Label('Income:'),
        dcc.Dropdown(
            id='Income',
            options=income_options,
            value=4  # Set default value
        ),
        html.Label('Education:'),
        dcc.Dropdown(
            id='Education',
            options=education_options,
            value=4  # Set default value
        ),
        html.Label('General Health:'),
        dcc.Dropdown(
            id='GenHlth',
            options=health_options,
            value=3  # Set default value
        ),
        html.Label('Physical Health:'),
        dcc.Dropdown(
            id='PhysHlth',
            options=health_options,
            value=3  # Set default value
        ),
        html.Label('Mental Health:'),
        dcc.Dropdown(
            id='MentHlth',
            options=health_options,
            value=3  # Set default value
        ),
        html.Label('High Blood Pressure:'),
        dcc.RadioItems(
            id='HighBP',
            options=[
                {'label': 'Yes', 'value': 1},
                {'label': 'No', 'value': 0}
            ],
            value=0
        ),
        html.Label('High Cholesterol:'),
        dcc.RadioItems(
            id='HighChol',
            options=[
                {'label': 'Yes', 'value': 1},
                {'label': 'No', 'value': 0}
            ],
            value=0
        ),
        html.Label('Cholesterol Check:'),
        dcc.RadioItems(
            id='CholCheck',
            options=[
                {'label': 'Yes', 'value': 1},
                {'label': 'No', 'value': 0}
            ],
            value=0
        ),
        html.Label('Smoker:'),
        dcc.RadioItems(
            id='Smoker',
            options=[
                {'label': 'Yes', 'value': 1},
                {'label': 'No', 'value': 0}
            ],
            value=0
        ),
        html.Label('Stroke:'),
        dcc.RadioItems(
            id='Stroke',
            options=[
                {'label': 'Yes', 'value': 1},
                {'label': 'No', 'value': 0}
            ],
            value=0
        ),
        html.Label('Heart Disease:'),
        dcc.RadioItems(
            id='HeartDiseaseorAttack',
            options=[
                {'label': 'Yes', 'value': 1},
                {'label': 'No', 'value': 0}
            ],
            value=0
        ),
        html.Label('Physically Active:'),
        dcc.RadioItems(
            id='PhysActivity',
            options=[
                {'label': 'Yes', 'value': 1},
                {'label': 'No', 'value': 0}
            ],
            value=0
        ),
        html.Label('Fruits Consumption:'),
        dcc.RadioItems(
            id='Fruits',
            options=[
                {'label': 'Yes', 'value': 1},
                {'label': 'No', 'value': 0}
            ],
            value=0
        ),
        html.Label('Veggies Consumption:'),
        dcc.RadioItems(
            id='Veggies',
            options=[
                {'label': 'Yes', 'value': 1},
                {'label': 'No', 'value': 0}
            ],
            value=0
        ),
        html.Label('Heavy Alcohol Consumption:'),
        dcc.RadioItems(
            id='HvyAlcoholConsump',
            options=[
                {'label': 'Yes', 'value': 1},
                {'label': 'No', 'value': 0}
            ],
            value=0
        ),
        html.Label('Healthcare Coverage:'),
        dcc.RadioItems(
            id='AnyHealthcare',
            options=[
                {'label': 'Yes', 'value': 1},
                {'label': 'No', 'value': 0}
            ],
            value=0
        ),
        html.Label('NoDoc:'),
        dcc.RadioItems(
            id='NoDocbcCost',
            options=[
                {'label': 'Yes', 'value': 1},
                {'label': 'No', 'value': 0}
            ],
            value=0
        ),
        html.Label('Difficulty Walking:'),
        dcc.RadioItems(
            id='DiffWalk',
            options=[
                {'label': 'Yes', 'value': 1},
                {'label': 'No', 'value': 0}
            ],
            value=0
        )

        # Add similar RadioItems for other binary inputs...
        # For brevity, let's assume the remaining inputs are added in a similar fashion
    ]),
    html.Button('Predict', id='predict-button', n_clicks=0),
    html.Div(id='output')
])

# Define callback to compute prediction based on user inputs
@callback(
    dash.dependencies.Output('output', 'children'),
    [dash.dependencies.Input('predict-button', 'n_clicks')],
    [dash.dependencies.State('HighBP', 'value'),
    dash.dependencies.State('HighChol', 'value'),
    dash.dependencies.State('CholCheck', 'value'),
    dash.dependencies.State('BMI', 'value'),
    dash.dependencies.State('Smoker', 'value'),
    dash.dependencies.State('Stroke', 'value'),
    dash.dependencies.State('HeartDiseaseorAttack', 'value'),
    dash.dependencies.State('PhysActivity', 'value'),
    dash.dependencies.State('Fruits', 'value'),
    dash.dependencies.State('Veggies', 'value'),
    dash.dependencies.State('HvyAlcoholConsump', 'value'),
    dash.dependencies.State('AnyHealthcare', 'value'),
    dash.dependencies.State('NoDocbcCost', 'value'),
    dash.dependencies.State('GenHlth', 'value'),
    dash.dependencies.State('MentHlth', 'value'),
    dash.dependencies.State('PhysHlth', 'value'),
    dash.dependencies.State('DiffWalk', 'value'),
    dash.dependencies.State('Sex', 'value'),
    dash.dependencies.State('Age', 'value'),
    dash.dependencies.State('Education', 'value'),
    dash.dependencies.State('Income', 'value')]
)
def predict(n_clicks, HighBP, HighChol, CholCheck, BMI, Smoker, Stroke, HeartDiseaseorAttack, PhysActivity, Fruits, Veggies, 
        HvyAlcoholConsump, AnyHealthcare, NoDocbcCost, GenHlth, MentHlth, PhysHlth, DiffWalk, Sex, Age, Education, Income):  # Include arguments for other variables
        # Check if the button was clicked
    if n_clicks > 0:
    
        # Create a NumPy array with the user inputs
        user_inputs = np.array([
            HighBP, HighChol, CholCheck, BMI, Smoker, Stroke, HeartDiseaseorAttack, PhysActivity, Fruits, Veggies, 
            HvyAlcoholConsump, AnyHealthcare, NoDocbcCost, GenHlth, MentHlth, PhysHlth, DiffWalk, Sex, Age, Education, Income        # Add other variables here in the same order they are provided in the model
            # For example: high_chol, chol_check, smoker, stroke, ...
        ]).reshape(1, -1)  # Reshape to match the expected input shape of the model

        scaled_inputs = scaler.transform(user_inputs)
        
        # Use the loaded logistic regression model to make predictions
        log_pred = log_reg.predict(scaled_inputs)
        log_prob = format(log_reg.predict_proba(scaled_inputs)[:,1][0]*100, '3f')
#        rand_prob = format(rand_for.predict_proba(scaled_inputs)[:,1][0]*100, '3f')

        return html.H2(f'Logistic Regression Prediction: {log_prob}% Probability of Diabetes.')
                       #Random Forest Classifier Prediction: {rand_prob}% Probability of Diabetes.')

    
