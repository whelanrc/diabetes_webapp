# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 15:53:15 2023

@author: raywh
"""

import dash
#import dash_core_components as dcc
#import dash_html_components as html
from dash import html, dcc
from dash.dependencies import Input, Output
#from Pages import Home, EDA, Modeling, Predict

#import dash_bootstrap_components as dbc

# Initialize the Dash app
app = dash.Dash(__name__, use_pages = True
                #,    external_stylesheets=[dbc.themes.BOOTSTRAP]
    )

# Define layout for the pages
app.layout = html.Div([
    html.H1('Multi-page app with Dash Pages'),
    html.Div([
        html.Div(
            dcc.Link(f"{page['name']} - {page['path']}", href=page["relative_path"])
        ) for page in dash.page_registry.values()
    ]),
    dash.page_container
])


if __name__ == '__main__':
    app.run_server(debug=True)
