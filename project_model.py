'''
Final Project of Data Science Bootcamp at Spiced Academy
Silke R. Schmidt, December 2022
Project: prediction of particulate matter in Berlin using traffic and weather data
--- MODEL DATA---
'''

#import os
#import glob
import numpy as np
import pandas as pd
#import seaborn as sns
#import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, PolynomialFeatures
#from sklearn.preprocessing import QuantileTransformer, PowerTransformer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
#from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, VotingRegressor

#
# read and prepare data
#
df = pd.read_pickle('data/df_marien_full.pkl')
df = df.dropna()

num_features = df.drop(columns=['Feinstaub (PM2,5)']).columns

X = df.drop(columns=['Feinstaub (PM2,5)'])
y = df['Feinstaub (PM2,5)']

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.25, random_state=42)

#
# Pipelines
#
num_transformer = Pipeline(
    steps=[
        ('scaler', MinMaxScaler())
        # ('scaler', StandardScaler())
        # preprocessing.PowerTransformer(method='box-cox', standardize=False)
    ])

column_transformer = ColumnTransformer(
    transformers=[
        ('num_trans', num_transformer, num_features)
    ], remainder='drop'
)

# Linear Regression
lm_pipeline = Pipeline(
    [
        ('col_trans', column_transformer),
        ('model', LinearRegression())
    ])
hyperparams_lm = {
    'col_trans__num_trans__scaler': [MinMaxScaler(), StandardScaler()],
}

# Polynomial Linear Regression
lm_poly_pipeline = Pipeline(
    [
        ('col_trans', column_transformer),
        ('polyFeat', PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)),
        ('model', LinearRegression())
    ])
hyperparams_lm_poly = {
    'col_trans__num_trans__scaler': [MinMaxScaler(), StandardScaler()],
    'polyFeat__interaction_only': [True, False],
    'polyFeat__degree': [1, 2, 3, 4, 5],
    'polyFeat__include_bias': [True, False]
}

# Random Forest
rf_pipeline = Pipeline(
    steps = [
        ('col_trans', column_transformer),
        ('model', RandomForestRegressor(
            n_estimators=300, max_depth=10, random_state=10, max_features='sqrt'))
])
hyperparams_rf = {
    #'preprocessor__pipeline-2__simpleimputer__strategy': ['mean', 'median'],
    'model__max_depth': [3, 4, 5, 10, 20, 50],
    'model__n_estimators': [10, 100, 200, 500, 1000],
    'model__max_features': ['sqrt', 'log2']
}

model_pipes = {
    'Linear Regression': lm_pipeline,
    'Polynomial Linear Regression': lm_poly_pipeline,
    'Random Forest': rf_pipeline
}
gridSearch_hyperparams = {
    'Linear Regression': hyperparams_lm,
    'Polynomial Linear Regression': hyperparams_lm_poly,
    'Random Forest': hyperparams_rf
}

model_names = model_pipes.keys()

def run_model(model_pipeline, dig=2):
    '''
    run model
    print train and test accuracy of model
    In case of GridSearch model: print best parameter set
    '''
    print(model_pipeline.upper())
    model_pipes[model_pipeline].fit(Xtrain, ytrain)
    ypred = model_pipes[model_pipeline].predict(Xtest)
    print('accuracy: ', 'train:', round(model_pipes[model_pipeline].score(Xtrain, ytrain), dig),
                        'test:', round(model_pipes[model_pipeline].score(Xtest, ytest), dig))
    print('-------------------------------------\n')

def run_gridSearch(model_pipeline, dig=2):
    '''run GridSearch on model pipeline'''
    print('GRIDSEARCH', model_pipeline.upper())
    gs_pipe = GridSearchCV(
        model_pipes[model_pipeline],
        gridSearch_hyperparams[model_pipeline],
        return_train_score=True,
        cv = 5,
        refit=True,
        verbose=True)
    gs_pipe.fit(Xtrain, ytrain)
    print('accuracy: ', 'train:', np.round(gs_pipe.cv_results_['mean_train_score'][0], dig),
                        'test:', np.round(gs_pipe.cv_results_['mean_test_score'][0], dig))
    print('best parameters', gs_pipe.best_params_)
    print('-------------------------------------\n')

for model in model_names:
    run_model(model)

for gs in model_names:
    run_gridSearch(gs)

# rf_feature_importance = pd.DataFrame(
#     zip(Xtrain.columns, rf_pipeline.named_steps['model'].feature_importances_),
#     columns=['feature', 'importance'])
# rf_feature_importance['abs_value'] = rf_feature_importance['importance'].apply(lambda x: abs(x))
# rf_feature_importance = rf_feature_importance.sort_values('abs_value', ascending=False)

# sns.set(rc={"figure.figsize":(12, 10)})
# _ = plt.figure()
# img = sns.barplot(y='feature', x='importance', data=rf_feature_importance)
# fig = img.get_figure()
# fig.savefig('plots/merged/Mariendorfer/RF_feature_importance.png', bbox_inches="tight")
# fig.clf()

# features_out_rf = rf_feature_importance[
#     rf_feature_importance['importance']<0.015]['feature'].values
