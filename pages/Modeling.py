# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 16:05:38 2023

@author: raywh
"""

import dash
#import dash_core_components as dcc
#import dash_html_components as html
from dash import Dash, html, dcc, callback, Output, Input
#from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score


dash.register_page(__name__)

#Data imports
filepath = 'Data/'

#Importing the binary diabetes data set
df = pd.read_csv(f'{filepath}diabetes_binary_health_indicators_BRFSS2015.csv')

pickle_path = 'PickleJar/'

# Load the pickled logistic regression model
with open(f'{pickle_path}/scaler.pkl', 'rb') as scale_file:
    scaler = pickle.load(scale_file)

with open(f'{pickle_path}/logistic_reg.pkl', 'rb') as log_file:
    log_reg = pickle.load(log_file)

#with open(f'{pickle_path}/random_forest.pkl', 'rb') as rand_file:
#    rand_for = pickle.load(rand_file)
    

models = {'Logistic Regression': log_reg}
#          'Random Forest': rand_for}

    

    
X = df.drop('Diabetes_binary', axis=1)  # Features
y = df['Diabetes_binary']  # Target variable

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

X_test = scaler.transform(X_test)


# Layout of the app
layout = html.Div([
    html.H1("Machine Learning Model Evaluation"),
    html.Label("Select a model:"),
    dcc.Dropdown(
        id='model-dropdown',
        options=[{'label': model_name, 'value': model_name} for model_name in models.keys()],
        value='Logistic Regression'
    ),
    html.Div(id='confusion-matrix'),
    html.Div(id='classification-report')
])

# Define callback to update confusion matrix and classification report
@callback(
    [Output('confusion-matrix', 'children'),
     Output('classification-report', 'children')],
    [Input('model-dropdown', 'value')]
)
def update_metrics(selected_model):
    model = models[selected_model]
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)

    cm_table = html.Div([
        html.H2('Confusion Matrix:'),
        html.Table([
            html.Thead(html.Tr([html.Th('')] + [html.Th(f'Predicted {i}') for i in range(len(cm[0]))])),
            html.Tbody([
                html.Tr([html.Th(f'Actual {i}')] + [html.Td(cm[i, j]) for j in range(len(cm[0]))]) for i in range(len(cm))
            ])
        ])
    ])

    cr_text = html.Div([
        html.H2('Classification Report:'),
        html.Pre(cr)
    ])

    return cm_table, cr_text