from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from nyoka import xgboost_to_pmml
from teradataml import create_context, remove_context
from teradataml.dataframe.dataframe import DataFrame
from aoa.stats import stats
from aoa.util.artefacts import save_plot

import joblib
import os

import numpy as np
import pandas as pd
import scipy
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import statsmodels.graphics.tsaplots as sgt
import statsmodels.tsa.stattools as sts
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima.arima import auto_arima
from pmdarima.arima import OCSBTest 
from statsmodels.tsa.arima_model import ARIMA
import seaborn as sns
#import yfinance
import warnings


def train(data_conf, model_conf, **kwargs):
    hyperparams = model_conf["hyperParameters"]

    create_context(host=os.environ["AOA_CONN_HOST"],
                   username=os.environ["AOA_CONN_USERNAME"],
                   password=os.environ["AOA_CONN_PASSWORD"])


    
    #feature_names = ["NumTimesPrg", "PlGlcConc", "BloodP", "SkinThick", "TwoHourSerIns", "BMI", "DiPedFunc", "Age"]
    #target_name = "HasDiabetes"

    # read training dataset from Teradata and convert to pandas
    #train_df = DataFrame(data_conf["table"])
    #train_df = train_df.select([feature_names + [target_name]])
    #train_df = train_df.to_pandas()

    #raw_data = yfinance.download (tickers = "VOW3.DE, PAH3.DE, BMW.DE", interval = "1d", group_by = 'ticker',
    #                          auto_adjust = True, treads = True)
    
    df = DataFrame(data_conf["table"])
    df = df.to_pandas()
    remove_context()
    df.set_index("DATESTAMP", inplace=True)

    print(df.head(1))
    print(df.tail(1))
    #df = raw_data.copy()
    
    # Hyperparameters
    
    # Starting Date
    #start_date = "2009-04-05" 
    start_date = hyperparams["start_date"]
    # First Official Announcement - 49.9%
    #ann_1 = "2009-12-09" 
    ann_1 = hyperparams["ann_1"]
    # Second Official Announcement - 51.1%
    #ann_2 = "2012-07-05" 
    ann_2 = hyperparams["ann_2"]
    #Ending Date
    #end_date = "2014-01-01"
    end_date = hyperparams["end_date"]
    # Dieselgate
    #d_gate = '2015-09-20' 
    d_gate = hyperparams["d_gate"]
    # Pre-processing the Data
    # Extracting Closing Prices
    df['vol'] = df['VOW3.DE_Close']
    df['por'] = df['PAH3.DE_Close']
    df['bmw'] = df['BMW.DE_Close']
    # Creating Returns
    df['ret_vol'] = df['vol'].pct_change(1).mul(100)
    df['ret_por'] = df['por'].pct_change(1).mul(100)
    df['ret_bmw'] = df['bmw'].pct_change(1).mul(100)
    # Extracting Volume
    df['vol'] = df['VOW3.DE_Close']
    df['por'] = df['PAH3.DE_Close']
    df['bmw'] = df['BMW.DE_Close']
    
    # split data into X and y
    #X_train = train_df.drop(target_name, 1)
    #y_train = train_df[target_name]

    print("Starting training...")

    mod_pr_pre_vol = auto_arima(df.vol[start_date:ann_1], exogenous = df[['por','bmw']][start_date:ann_1],
                            m = hyperparams["m"], max_p = hyperparams["max_p"], max_q = hyperparams["max_q"])

    # fit model to training data
    #model = Pipeline([('scaler', MinMaxScaler()),
    #                  ('xgb', XGBClassifier(eta=hyperparams["eta"],
    #                                        max_depth=hyperparams["max_depth"]))])
    
    
    # xgboost saves feature names but lets store on pipeline for easy access later
    mod_pr_pre_vol.feature_names = ['por','bmw']
    mod_pr_pre_vol.target_name = 'vol'

    #model.fit(X_train, y_train)

    print("Finished training")

    # export model artefacts
    joblib.dump(mod_pr_pre_vol, "artifacts/output/model.joblib")

    # we can also save as pmml so it can be used for In-Vantage scoring etc.
    #xgboost_to_pmml(pipeline=model, col_names=feature_names, target_name=target_name, pmml_f_name="artifacts/output/model.pmml")

    print("Saved trained model")
    
    #from xgboost import plot_importance
    #model["xgb"].get_booster().feature_names = feature_names
    #plot_importance(model["xgb"].get_booster(), max_num_features=10)
    #save_plot("feature_importance.png")

    mod_pr_pre_vol.plot_diagnostics(figsize=(15,15));
    save_plot("diagnostics.png")
    
    #feature_importance = model["xgb"].get_booster().get_score(importance_type="weight")
    #stats.record_training_stats(train_df,
    #                   features=feature_names,
    #                   predictors=[target_name],
    #                   categorical=[target_name],
    #                   importance=feature_importance,
    #                   category_labels={target_name: {0: "false", 1: "true"}})