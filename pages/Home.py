# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 16:01:09 2023

@author: raywh
"""
import dash
from dash import Dash, html, dcc, callback, Output, Input

dash.register_page(__name__, path='/')

layout = html.Div([
    html.H1('This is our Home page'),
    html.Div('This is our Home page content.'),
])
