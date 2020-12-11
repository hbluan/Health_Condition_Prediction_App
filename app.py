#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 15:37:17 2020

@author: simon
"""

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import dash_table
from plotly.subplots import make_subplots


import pandas as pd
import numpy as np
import os
import calendar
from pathlib import Path
import networkx as nx
import seaborn as sns
import decimal


from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, scale
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, cross_validate, ShuffleSplit, train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import plot_tree
from sklearn.svm import LinearSVC, SVC

from sklearn.linear_model import LinearRegression, Lasso, Ridge, LassoCV, BayesianRidge
import statsmodels.formula.api as sm

from dmba import regressionSummary, exhaustive_search
from dmba import backward_elimination, forward_selection, stepwise_selection
from dmba import adjusted_r2_score, AIC_score, BIC_score
from dmba import plotDecisionTree, classificationSummary, regressionSummary

import mglearn
import pydotplus

import warnings
warnings.filterwarnings('ignore')

from model.NHANES_prediction import clean_and_split_data, calculate_model, health_summary


#%%
def max_index(prob_summary):
    max_index = 0
    for i in range(len(prob_summary)):
        if prob_summary[i] == max(prob_summary):
            max_index = i
    return max_index

def get_picture(condition):
    return "/assets/{}.png".format(condition)


#%% Load Data
health_df = pd.read_csv('Data/NHANES.csv')

X, y_depression, y_frax, y_hypertension, y_cholesterol, y_diabetes, y_oral = clean_and_split_data(health_df)

nb_depression, Forest_frax, Forest_hypertension, Forest_cholesterol, nb_diabetes, cv_svm_oral = calculate_model(X, y_depression, y_frax, y_hypertension, y_cholesterol, y_diabetes, y_oral)

# health_data_1 = [3, 2, 4, 93.4, 166.6, 33.7, 1, 0, 5, 3, 2.0, 1, 0, 0, 0, 1, 10.0, 7, 0, 1.0, 250.0, 98.6, 150.0]

# pred_summary, prob_summary = health_summary(np.array(health_data_1).reshape(-1, 23))

#%% Initial Settings

age_group_list = [('20-29',2),('30-39',3),('40-49',4),('50-59',5),('60-79',6),('70-79',7),('80+',8)]

race_list = [('Mexican American',1),('Other Hispanic',2),('Non-Hispanic White',3),('Non-Hispanic Black',4),('Non-Hispanic Asian',6),('Other Race - Including Multi-Racial',7)]

household_size_list= [('1',1),('2',2),('3',3),('4',4),('5',5),('6',6),('7 or more',7)]

health_condition_list= [('Excellent',1),('Very Good',2),('Good',3),('Fair',4),('Poor',5)]

condition_list = ['Depression', 'Fraxture', 'Hypertension', 'High Cholesterol', 'Diabetes', 'Dental Health']

prob_summary_initial = (0.5, 0.5, 0.5, 0.5, 0.5, 0.5)

# width =  800
# height = 400

#%% Build Dash
app = dash.Dash(external_stylesheets=[dbc.themes.SANDSTONE])

control_1 = dbc.Card(
    [
        dbc.FormGroup(
            [
                dbc.Label("Age Group"),
                dcc.Dropdown(
                    options=[{"label": age_group, "value": age_group_value} for age_group, age_group_value in age_group_list],
                    id="age_group",
                    value = 2
                ),
            ]
        ),
        dbc.FormGroup(
            [
                dbc.Label("Gender"),
                dcc.RadioItems(
                    options=[{"label": gender, "value": gender_value} for gender, gender_value in [('Male',1),('Female',2)]],
                    id="gender"
                ),
            ]
        ),
        dbc.FormGroup(
            [
                dbc.Label("Race"),
                dcc.Dropdown(
                    options=[{"label": race, "value": race_value} for race, race_value in race_list],
                    id="race",
                    value = 1
                ),
            ]
        ),
        dbc.FormGroup(
            [
                dbc.Label("Height in centimeter"),
                dcc.Input(
                    id='height',
                    placeholder='Enter height in centimeter',
                    type='number',
                    min=1,
                    value = 180
                    ),
            ]
        ),
        dbc.FormGroup(
            [
                dbc.Label("Weight in kilogram"),
                dcc.Input(
                    id='weight',
                    placeholder='Enter weight in kilogram',
                    type='number',
                    min=1,
                    value = 80
                    ),
            ]
        ),
    ],
    body=True,
)



control_2 = dbc.Card(
    [
        dbc.FormGroup(
            [
                dbc.Label("Highest education"),
                dcc.Dropdown(
                    options=[{"label": education, "value": education_value} for education, education_value in [('High School or under',0),('College or above',1)]],
                    id="college",
                    value = 1
                ),
            ]
        ),
        dbc.FormGroup(
            [
                dbc.Label("Drink milk at least every week"),
                dcc.RadioItems(
                    options=[{"label": drink_milk, "value": drink_milk_value} for drink_milk, drink_milk_value in [('Yes',1),('No',0)]],
                    id="drink_milk",
                    value = 1
                ),
            ]
        ),
        dbc.FormGroup(
            [
                dbc.Label("Number of people in your household"),
                dcc.Dropdown(
                    options=[{"label": household_size, "value": household_size_value} for household_size, household_size_value in household_size_list],
                    id="household_size",
                    value = 5
                ),
            ]
        ),
        dbc.FormGroup(
            [
                dbc.Label("How would you evaluate your current health condition "),
                dcc.Dropdown(
                    options=[{"label": health_condition, "value": health_condition_value} for health_condition, health_condition_value in health_condition_list],
                    id="health_condition",
                    value = 3
                ),
            ]
        ),
        dbc.FormGroup(
            [
                dbc.Label("Number of fast food had in the past seven days"),
                dcc.Input(
                    id='fast_food',
                    placeholder='Enter 22 if you have more than 22 fast food',
                    type='number',
                    min= 0,
                    max = 22,
                    value = 0
                ),
            ]
        )
    ],
    body=True,
)





control_3 = dbc.Card(
    [
         dbc.FormGroup(
            [
                dbc.Label("Use restaurant nutrition info when dining"),
                dcc.RadioItems(
                    options=[{"label": nutrition_info, "value": nutrition_info_value} for nutrition_info, nutrition_info_value in [('Yes',1),('No',0)]],
                    id="nutrition_info",
                    value = 1
                ),
            ]
        ),
        dbc.FormGroup(
            [
                dbc.Label("Check blood pressure at last every year"),
                dcc.RadioItems(
                    options=[{"label": pressure_check, "value": pressure_check_value} for pressure_check, pressure_check_value in [('Yes',1),('No',0)]],
                    id="pressure_check",
                    value = 1
                ),
            ]
        ),
        dbc.FormGroup(
            [
                dbc.Label("Have family history of heart attack"),
                dcc.RadioItems(
                    options=[{"label": family_heart, "value": family_heart_value} for family_heart, family_heart_value in [('Yes',1),('No',0)]],
                    id="family_heart",
                    value = 1
                ),
            ]
        ),
        dbc.FormGroup(
            [
                dbc.Label("Have family history of diabetes"),
                dcc.RadioItems(
                    options=[{"label": family_diabetes, "value": family_diabetes_value} for family_diabetes, family_diabetes_value in [('Yes',1),('No',0)]],
                    id="family_diabetes",
                    value = 1
                ),
            ]
        ),
        dbc.FormGroup(
            [
                dbc.Label("Have family history of osteoporosis"),
                dcc.RadioItems(
                    options=[{"label": family_osteoporosis, "value": family_osteoporosis_value} for family_osteoporosis, family_osteoporosis_value in [('Yes',1),('No',0)]],
                    id="family_osteoporosis",
                    value = 0
                ),
            ]
        )
    ],
    body=True,
)



control_4 = dbc.Card(
    [
         dbc.FormGroup(
            [
                dbc.Label("Number of hours spend sitting on a typical day"),
                dcc.Input(
                    id='sit_hour',
                    placeholder='Enter 0-24',
                    type='number',
                    min = 0,
                    max = 24,
                    value = 8
                    ),
            ]
        ),
        dbc.FormGroup(
            [
                dbc.Label("Number of hours sleeping on a typical day"),
                dcc.Input(
                    id='sleep_hour',
                    placeholder='Enter 0-24',
                    type='number',
                    min = 0,
                    max = 24,
                    value = 8
                    ),
            ]
        ),
        dbc.FormGroup(
            [
                dbc.Label("Have smoked at least 100 cigarettes in life"),
                dcc.RadioItems(
                    options=[{"label": smoke_100, "value": smoke_100_value} for smoke_100, smoke_100_value in [('Yes',1),('No',0)]],
                    id="smoke_100",
                    value = 0
                ),
            ]
        ),
        dbc.FormGroup(
            [
                dbc.Label("Number of days drink alcohol in the past twelve months"),
                dcc.Input(
                    id='drink_days',
                    placeholder='Enter 0-365',
                    type='number',
                    min = 0,
                    max = 365,
                    value = 30
                    ),
            ]
        )
    ],
    body=True,
)


control_5 = dbc.Card(
    [
        dbc.FormGroup(
            [
                dbc.Label("Monthly food expenditure (exclude restaurant and deliver)"),
                dcc.Input(
                    id='cost_food',
                    placeholder='Enter dollar amount',
                    type='number',
                    min = 0,
                    value = 400
                    ),
            ]
        ),
        dbc.FormGroup(
            [
                dbc.Label("Monthly non-food expenditure"),
                dcc.Input(
                    id='cost_nonfood',
                    placeholder='Enter dollar amount',
                    type='number',
                    min = 0,
                    value = 400
                    )
            ]
        ),
        dbc.FormGroup(
            [
                dbc.Label("Monthly restaurant expenditure"),
                dbc.Label("Please include delivery expenditure during COVID-19 lockdown"),
                dcc.Input(
                    id='cost_restaurant',
                    placeholder='Enter dollar amount',
                    type='number',
                    min = 0,
                    value = 400
                    )
            ]
        ),
        dbc.Button(id = 'submit',n_clicks = 0, children = "Submit", outline=True, color="primary", className="mr-1"),
        html.Div(id='input-check', children=[])
    ],
    body=True,
)




colors = {
    'text':'#095CE3',
    'plot_color': '#D1DBE8',
    'paper_color':'#FFFFFF'
}

app.layout = dbc.Container(
    [
        dbc.Row(
            dbc.Col(
                html.H1("Health Condition Prediction App", style={'text-align': 'center'})
            )
        ),
        
        dbc.Row(
            children = [
                dbc.Col(control_1, md = 2),
                dbc.Col(control_2, md = 2),
                dbc.Col(control_3, md = 2),
                dbc.Col(control_4, md = 2),
                dbc.Col(control_5, md = 2)
            ]
        ),
        dbc.Row(
            children = [
                dbc.Col(
                    [
                        dcc.Graph(
                            id = 'prob_plot',
                            figure = {
                                'data':[
                                    {'x':condition_list,'y':prob_summary_initial,'type':'bar', 'name':'Probability Summary'}
                                    ],
                                'layout':{
                                    'plot_bgcolor': colors['plot_color'],
                                    'paper_bgcolor': colors['paper_color'],
                                    'font':{
                                        'color': colors['text']
                                        },
                                    'title': '<b>Probability Summary</b>',
                                    #'width': width, 
                                    #'height': height
                                    }       
                                }
                            )
                        ],
                    md = 5
                ),
                dbc.Col(
                    html.Img(
                        id='image',
                        src=get_picture('Health'), 
                        style={'height':'80%', 'width':'80%'}
                        ), 
                    md=5)
                ]   
        )
        
    ],
    id="main-container",
    style={"display": "flex", "flex-direction": "column"},
    fluid=True
)
@app.callback(
    [Output(component_id='input-check', component_property='children'),
     Output(component_id='prob_plot', component_property='figure'),
     Output(component_id='image', component_property='src')],
    
    Input(component_id='submit',component_property='n_clicks'),
    
    [State(component_id='age_group', component_property='value'),
     State(component_id='gender', component_property='value'),
     State(component_id='race', component_property='value'),
     State(component_id='height', component_property='value'),
     State(component_id='weight', component_property='value'),
     State(component_id='college', component_property='value'),
     State(component_id='drink_milk', component_property='value'),
     State(component_id='household_size', component_property='value'),
     State(component_id='health_condition', component_property='value'),
     State(component_id='fast_food', component_property='value'),
     State(component_id='nutrition_info', component_property='value'),
     State(component_id='pressure_check', component_property='value'),
     State(component_id='family_heart', component_property='value'),
     State(component_id='family_diabetes', component_property='value'),
     State(component_id='family_osteoporosis', component_property='value'),
     State(component_id='sit_hour', component_property='value'),
     State(component_id='sleep_hour', component_property='value'),
     State(component_id='smoke_100', component_property='value'),
     State(component_id='drink_days', component_property='value'),
     State(component_id='cost_food', component_property='value'),
     State(component_id='cost_nonfood', component_property='value'),
     State(component_id='cost_restaurant', component_property='value')]
)
def get_weight(n_clicks, age_group, gender, race, weight, height, college, drink_milk, household_size, health_condition, fast_food, nutrition_info, pressure_check, family_heart, family_diabetes, family_osteoporosis, sit_hour, sleep_hour, smoke_100, drink_days, cost_food, cost_nonfood, cost_restaurant):
    try:
        input_valid_test = age_group * gender * race * height * weight * college * drink_milk * household_size * health_condition * fast_food * nutrition_info * pressure_check * family_heart * family_diabetes * family_osteoporosis * sit_hour * sleep_hour * smoke_100 * drink_days * cost_food * cost_nonfood * cost_restaurant
        
        feed_back = ["All data is valid "]
        
        bmi = weight / ((height/100)**2)
        user_input = [age_group, gender, race, weight, height, bmi, college, drink_milk, household_size, health_condition, fast_food, nutrition_info, pressure_check, family_heart, family_diabetes, family_osteoporosis, sit_hour, sleep_hour, smoke_100, drink_days, cost_food, cost_nonfood, cost_restaurant]

        pred_summary, prob_summary = health_summary(np.array(user_input).reshape(-1, 23))
        
        figure = {
            'data':[
                {'x':condition_list,'y':prob_summary,'type':'bar', 'name':'Probability Summary'}
            ],
            'layout':{
                'plot_bgcolor': colors['plot_color'],
                'paper_bgcolor': colors['paper_color'],
                'font':{
                    'color': colors['text']
                },
                'yaxis': {'range': [0, 1]},
                'title': '<b>Probability Summary</b>'
                #'width': width, 
                #'height': height
            }
        }
        # Find the condition with the highest probability, and get the path of that picture
        predicted_condition_index = max_index(prob_summary)
        picture_path = get_picture(condition_list[predicted_condition_index])
        
    
    except TypeError:
        figure = {
            'data':[
                {'x':condition_list,'y':prob_summary_initial,'type':'bar', 'name':'Probability Summary'}
            ],
            'layout':{
                'plot_bgcolor': colors['plot_color'],
                'paper_bgcolor': colors['paper_color'],
                'font':{
                    'color': colors['text']
                },
                'yaxis': {'range': [0, 1]},
                'title': '<b>Probability Summary</b>'
                #'width': width, 
                #'height': height
            }
        }
        feed_back = ["Please fill all input before submition"]
        
        picture_path = get_picture('Health')

    return feed_back, figure, picture_path

#%%

# Main
if __name__ == "__main__":
    app.run_server(debug=True)




















































