#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 14:58:18 2020

@author: simon
"""

import numpy as np
import pandas as pd

import os
import calendar
from pathlib import Path
import networkx as nx
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.pylab as plt
from pandas.plotting import scatter_matrix, parallel_coordinates

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


from IPython.display import display

#%%
def max_index(prob_summary):
    max_index = 0
    for i in range(len(prob_summary)):
        if prob_summary[i] == max(prob_summary):
            max_index = i
    return max_index



#%% 1. Load and clean data
health_df = pd.read_csv('Data/NHANES.csv')

categorical_list = ("age_group", "gender", "race", "edu", "college", "drink_milk", "home_owned", "health_condition", "nutrition_info", "heavy_sport", "family_heart", "family_diabetes", "family_osteoporosis", "smoke_now", "smoke_100", "poverty_level_category", "depression", "frax_risk", "hypertension", "cholesterol", "diabetes", "oral")

for var in categorical_list:
    health_df[var] = health_df[var].astype('category',copy=False)
    
selected_list = ['age_group', 'gender', 'race', 'weight', 'height', 'bmi', 'college', 'drink_milk', 
                 'household_size', 'health_condition', 'fast_food', 'nutrition_info', 'pressure_check', 
                 'family_heart', 'family_diabetes', 'family_osteoporosis', 'sit_hour', 'sleep_hour', 
                 'smoke_100', 'drink_days', 'cost_food', 'cost_nonfood', 'cost_restaurant',
                'depression', 'frax_risk', 'hypertension', 'cholesterol', 'diabetes', 'oral']

health_df_selected = health_df[selected_list]

print(health_df_selected.shape)

#%%n 2. Data Spliting
health_df_selected['random'] = np.random.normal(size=5769 ,loc=0)

health_df_selected = health_df_selected.sort_values(by = ['random'])
health_df_train_test = health_df_selected.iloc[:5000,:]
health_df_valid = health_df_selected.iloc[5000:,:]

print(health_df_train_test.shape)
print(health_df_valid.shape)


X = health_df_train_test.drop(['depression', 'frax_risk', 'hypertension', 'cholesterol', 'diabetes', 'oral','random'],axis=1)
y_depression = health_df_train_test.depression
y_frax = health_df_train_test.frax_risk
y_hypertension = health_df_train_test.hypertension
y_cholesterol = health_df_train_test.cholesterol
y_diabetes = health_df_train_test.diabetes
y_oral = health_df_train_test.oral


X_v = health_df_valid.drop(['depression', 'frax_risk', 'hypertension', 'cholesterol', 'diabetes', 'oral','random'],axis=1)
y_depression_v = health_df_valid.depression
y_frax_v = health_df_valid.frax_risk
y_hypertension_v = health_df_valid.hypertension
y_cholesterol_v = health_df_valid.cholesterol
y_diabetes_v = health_df_valid.diabetes
y_oral_v = health_df_valid.oral

#%% 3.Optimal Models
# shuffle = ShuffleSplit(n_splits=100, test_size=.2, random_state=1)

health_data = X_v.iloc[1,:].values.reshape(-1, 23)
#type(np.array(health_data_1))
    # 3.1 Depression - Naive Bayes
X_train, X_test, y_train, y_test = train_test_split(X, y_depression, random_state=1, test_size = 0.2)

nbm = MultinomialNB(alpha=1000)
nb_depression=nbm.fit(X_train,y_train)

depression_pred = nb_depression.predict(health_data)
depression_prob = nb_depression.predict_proba(health_data)[:,1]

    #%% 3.2 Frax - Random Forest
X_train, X_test, y_train, y_test = train_test_split(X, y_frax, random_state=1, test_size = 0.2)

forest_frax = RandomForestClassifier(n_estimators=50,max_features=6,max_depth=15)
Forest_frax=forest_frax.fit(X_train, y_train)

frax_pred = Forest_frax.predict(health_data)
frax_prob = Forest_frax.predict_proba(health_data)[:,1]

    #%% 3.3 Hypertension - Random Forest
X_train, X_test, y_train, y_test = train_test_split(X, y_hypertension, random_state=1, test_size = 0.2)

forest_hypertension = RandomForestClassifier(n_estimators=150,max_features=5,max_depth=5)
Forest_hypertension=forest_hypertension.fit(X_train, y_train)

hypertension_pred = Forest_hypertension.predict(health_data)
hypertension_prob = Forest_hypertension.predict_proba(health_data)[:,1]


    #%% 3.4 Cholesterol - Random Forest
X_train, X_test, y_train, y_test = train_test_split(X, y_cholesterol, random_state=1, test_size = 0.2)

forest_cholesterol = RandomForestClassifier(n_estimators=50,max_features=6,max_depth=5)
Forest_cholesterol=forest_cholesterol.fit(X_train, y_train)

cholesterol_pred = Forest_cholesterol.predict(health_data)
cholesterol_prob = Forest_cholesterol.predict_proba(health_data)[:,1]


    #%% 3.5 Diabetes - Naive Bayes
X_train, X_test, y_train, y_test = train_test_split(X, y_diabetes, random_state=1, test_size = 0.2)

nbm = MultinomialNB(alpha=10)
nb_diabetes=nbm.fit(X_train,y_train)

diabetes_pred = nb_diabetes.predict(health_data)
diabetes_prob = nb_diabetes.predict_proba(health_data)[:,1]


    #%% 3.6 Oral - SVM
X_train, X_test, y_train, y_test = train_test_split(X, y_oral, random_state=1, test_size = 0.2)

steps = [('scaler', StandardScaler()),
         ('svm', SVC(probability = True))]

pipeline = Pipeline(steps)
parameters = {'svm__C':[1],
              'svm__gamma':[0.01]}

cv_svm_oral = GridSearchCV(pipeline, param_grid=parameters,cv=5)
cv_svm_oral.fit(X_train, y_train)
oral_pred = cv_svm_oral.predict(health_data)

oral_prob = cv_svm_oral.predict_proba(health_data)[:,1]

#%% Probability Matrix
prob = (float(depression_prob), float(frax_prob), hypertension_prob, cholesterol_prob, diabetes_prob, oral_prob)



#%% Function to build dash
def clean_and_split_data(health_df):
    categorical_list = ("age_group", "gender", "race", "edu", "college", "drink_milk", "home_owned", "health_condition", "nutrition_info", "heavy_sport", "family_heart", "family_diabetes", "family_osteoporosis", "smoke_now", "smoke_100", "poverty_level_category", "depression", "frax_risk", "hypertension", "cholesterol", "diabetes", "oral")
    
    for var in categorical_list:
        health_df[var] = health_df[var].astype('category',copy=False)
        
    selected_list = ['age_group', 'gender', 'race', 'weight', 'height', 'bmi', 'college', 'drink_milk', 
                     'household_size', 'health_condition', 'fast_food', 'nutrition_info', 'pressure_check', 
                     'family_heart', 'family_diabetes', 'family_osteoporosis', 'sit_hour', 'sleep_hour', 
                     'smoke_100', 'drink_days', 'cost_food', 'cost_nonfood', 'cost_restaurant',
                    'depression', 'frax_risk', 'hypertension', 'cholesterol', 'diabetes', 'oral']
    
    health_df_selected = health_df[selected_list]
    
    health_df_selected['random'] = np.random.normal(size=5769 ,loc=0)

    health_df_selected = health_df_selected.sort_values(by = ['random'])
    health_df_train_test = health_df_selected.iloc[:5000,:]
    
    X = health_df_train_test.drop(['depression', 'frax_risk', 'hypertension', 'cholesterol', 'diabetes', 'oral','random'],axis=1)
    y_depression = health_df_train_test.depression
    y_frax = health_df_train_test.frax_risk
    y_hypertension = health_df_train_test.hypertension
    y_cholesterol = health_df_train_test.cholesterol
    y_diabetes = health_df_train_test.diabetes
    y_oral = health_df_train_test.oral
    
    return X, y_depression, y_frax, y_hypertension, y_cholesterol, y_diabetes, y_oral


def calculate_model(X, y_depression, y_frax, y_hypertension, y_cholesterol, y_diabetes, y_oral):
    # 1 Depression - Naive Bayes
    X_train, X_test, y_train, y_test = train_test_split(X, y_depression, random_state=1, test_size = 0.2)  
    nbm = MultinomialNB(alpha=1000)
    nb_depression=nbm.fit(X_train,y_train)
    
    # 2 Frax - Random Forest
    X_train, X_test, y_train, y_test = train_test_split(X, y_frax, random_state=1, test_size = 0.2)
    forest_frax = RandomForestClassifier(n_estimators=50,max_features=6,max_depth=15)
    Forest_frax=forest_frax.fit(X_train, y_train)
    
    # 3 Hypertension - Random Forest
    X_train, X_test, y_train, y_test = train_test_split(X, y_hypertension, random_state=1, test_size = 0.2)
    forest_hypertension = RandomForestClassifier(n_estimators=150,max_features=5,max_depth=5)
    Forest_hypertension=forest_hypertension.fit(X_train, y_train)
    
    # 4 Cholesterol - Random Forest
    X_train, X_test, y_train, y_test = train_test_split(X, y_cholesterol, random_state=1, test_size = 0.2)
    forest_cholesterol = RandomForestClassifier(n_estimators=50,max_features=6,max_depth=5)
    Forest_cholesterol=forest_cholesterol.fit(X_train, y_train)
    
    # 5 Diabetes - Naive Bayes
    X_train, X_test, y_train, y_test = train_test_split(X, y_diabetes, random_state=1, test_size = 0.2)
    nbm = MultinomialNB(alpha=10)
    nb_diabetes=nbm.fit(X_train,y_train)
    
    # 6 Oral - SVM
    X_train, X_test, y_train, y_test = train_test_split(X, y_oral, random_state=1, test_size = 0.2)
    
    steps = [('scaler', StandardScaler()),
             ('svm', SVC(probability = True))]
    
    pipeline = Pipeline(steps)
    parameters = {'svm__C':[1],
                  'svm__gamma':[0.01]}
    
    cv_svm_oral = GridSearchCV(pipeline, param_grid=parameters,cv=5)
    cv_svm_oral.fit(X_train, y_train)
    
    return nb_depression, Forest_frax, Forest_hypertension, Forest_cholesterol, nb_diabetes, cv_svm_oral



def health_summary(health_data):
    depression_pred_value       = int(nb_depression.predict(health_data))
    depression_prob_value       = round(float(nb_depression.predict_proba(health_data)[:,1]),4)
    depression_prob_percent     = "{:.2%}".format(depression_prob_value)
    
    frax_pred_value             = int(Forest_frax.predict(health_data))
    frax_prob_value             = round(float(Forest_frax.predict_proba(health_data)[:,1]),4)
    frax_prob_percent           = "{:.2%}".format(frax_prob_value)
    
    hypertension_pred_value     = int(Forest_hypertension.predict(health_data))
    hypertension_prob_value     = round(float(Forest_hypertension.predict_proba(health_data)[:,1]),4)
    hypertension_prob_percent   = "{:.2%}".format(hypertension_prob_value)

    cholesterol_pred_value      = int(Forest_cholesterol.predict(health_data))
    cholesterol_prob_value      = round(float(Forest_cholesterol.predict_proba(health_data)[:,1]),4)
    cholesterol_prob_percent    = "{:.2%}".format(cholesterol_prob_value)

    diabetes_pred_value         = int(nb_diabetes.predict(health_data))
    diabetes_prob_value         = round(float(nb_diabetes.predict_proba(health_data)[:,1]),4)
    diabetes_prob_percent       = "{:.2%}".format(diabetes_prob_value)
    
    oral_pred_value             = int(cv_svm_oral.predict(health_data))
    oral_prob_value             = round(float(cv_svm_oral.predict_proba(health_data)[:,1]),4)
    oral_prob_percent           = "{:.2%}".format(oral_prob_value)

    pred_summary = (depression_pred_value, frax_pred_value, hypertension_pred_value, cholesterol_pred_value, diabetes_pred_value, oral_pred_value)
    prob_summary = (depression_prob_value, frax_prob_value, hypertension_prob_value, cholesterol_prob_value, diabetes_prob_value, oral_prob_value)

    return pred_summary, prob_summary













