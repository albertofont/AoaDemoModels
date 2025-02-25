from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from nyoka import xgboost_to_pmml
from teradataml import create_context, remove_context
from teradataml.dataframe.dataframe import DataFrame
from aoa.stats import stats
from aoa.util.artefacts import save_plot
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.graphics.tsaplots import month_plot,quarter_plot
from statsmodels.tsa.seasonal import seasonal_decompose
import sys
from io import StringIO


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
from statsmodels.tsa.stattools import adfuller
import seaborn as sns
#import yfinance
import warnings


def adf_test(series,title=''):
    """
    Pass in a time series and an optional title, returns an ADF report
    """
    print(f'Augmented Dickey-Fuller Test: {title}')
    result = adfuller(series.dropna(),autolag='AIC') # .dropna() handles differenced data
    
    labels = ['ADF test statistic','p-value','# lags used','# observations']
    out = pd.Series(result[0:4],index=labels)

    for key,val in result[4].items():
        out[f'critical value ({key})']=val
        
    print(out.to_string())          # .to_string() removes the line "dtype: float64"
    
    if result[1] <= 0.05:
        print("Strong evidence against the null hypothesis")
        print("Reject the null hypothesis")
        print("Data has no unit root and is stationary")
    else:
        print("Weak evidence against the null hypothesis")
        print("Fail to reject the null hypothesis")
        print("Data has a unit root and is non-stationary")

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
    # Assigning the Frequency and Filling NA Values
    df = df.asfreq('b')
    df = df.fillna(method='bfill')
    
    # split data into X and y
    #X_train = train_df.drop(target_name, 1)
    #y_train = train_df[target_name]

    print("Performiing DF Test...")
    stdout_backup = sys.stdout
    sys.stdout = string_buffer = StringIO()
    adf_test(df['vol'])
    with open("artifacts/output/DFtest.txt", "w") as my_file:
        my_file.write(string_buffer.getvalue())    # write a line to the file
    sys.stdout = stdout_backup 
    print("Finished test")
    
    print("Starting training...")
    stdout_backup = sys.stdout
    sys.stdout = string_buffer = StringIO()
    mod_pr_pre_vol = auto_arima(df.vol[start_date:ann_1], exogenous = df[['por','bmw']][start_date:ann_1],
                            m = hyperparams["m"], max_p = hyperparams["max_p"], max_q = hyperparams["max_q"],
                            d=None, trace=True,
                            error_action='ignore',   
                            suppress_warnings=True,  
                            stepwise=True)

    # fit model to training data
    #model = Pipeline([('scaler', MinMaxScaler()),
    #                  ('xgb', XGBClassifier(eta=hyperparams["eta"],
    #                                        max_depth=hyperparams["max_depth"]))])
    with open("artifacts/output/summary.txt", "w") as my_file:
        my_file.write(string_buffer.getvalue())    # write a line to the file
    sys.stdout = stdout_backup 
    
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
    
    df['vol'][start_date:ann_1].plot(figsize= (20,8), color = "#3386FF")
    df['por'][start_date:ann_1].plot(color = "#33FF8F")
    df['bmw'][start_date:ann_1].plot(color = "#DFD22D")

    df['vol'][ann_1:ann_2].plot(color = "#2665BF")
    df['por'][ann_1:ann_2].plot(color = "#26BF6B")
    df['bmw'][ann_1:ann_2].plot(color = "#9F9620")

    df['vol'][ann_2:end_date].plot(color = "#1A4380")
    df['por'][ann_2:end_date].plot(color = "#1A8048")
    df['bmw'][ann_2:end_date].plot(color = "#605A13")

    df['vol'][end_date:].plot(color = "#0D2240")
    df['por'][end_date:].plot(color = "#0D4024")
    df['bmw'][end_date:].plot(color = "#403C0D")

    plt.legend(['Volkswagen','Porsche','BMW'])

    save_plot("Stocks_VOL_POR_BMW.png")
    
    title = 'Autocorrelation_Stocks_VOL'
    lags = 40
    plot_acf(df['vol'],lags=lags);
    
    save_plot("Autocorrelation_Stocks_VOL.png")

    title = 'PartialAutocorrelation_Stocks_VOL'
    lags = 40
    plot_acf(df['vol'],lags=lags);
    
    save_plot("PartialAutocorrelation_Stocks_VOL.png")
        
    plt.rc('figure', figsize=(12, 7))
    #plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old approach
    plt.text(0.01, 0.05, str(mod_pr_pre_vol.summary()), {'fontsize': 12}, fontproperties = 'monospace') # approach improved by OP -> monospace!
    plt.axis('off')
    plt.tight_layout()
    save_plot('Model_Summary')
    #plt.savefig('output.png')
    
    dfq = df['vol'].resample(rule='Q').mean()
    quarter_plot(dfq);
    save_plot('Quarter_plot')

    
    result = seasonal_decompose(df['vol'], model='additive')  # model='add' also works
    result.plot();
    save_plot('Sesasonal_decompose')
    
    #feature_importance = model["xgb"].get_booster().get_score(importance_type="weight")
    #stats.record_training_stats(train_df,
    #                   features=feature_names,
    #                   predictors=[target_name],
    #                   categorical=[target_name],
    #                   importance=feature_importance,
    #                   category_labels={target_name: {0: "false", 1: "true"}})
    
    #import shap
    #mapper = model['mapper']
    #shap_explainer = shap.TreeExplainer(model['regressor'])
    #X_train = pd.DataFrame(mapper.transform(X_train), columns=model.feature_names_tr)
    #X_shap = shap.sample(X_train, 100)
    #shap_values = shap_explainer.shap_values(X_shap)
    #feature_importances = pd.DataFrame(list(zip(model.feature_names_tr,
    #                                         np.abs(shap_values).mean(0))),
    #                               columns=['col_name', 'feature_importance_vals'])
    #feature_importances = feature_importances.set_index("col_name").T.to_dict(orient='records')[0]
    
    #category_labels_overrides = {
    #    "emailer_for_promotion": {0: "false", 1: "true"},
    #    "homepage_featured": {0: "Not featured", 1: "Featured"}
    #}
    
    #mod_pr_pre_vol.categorical=[]
    #stats.record_training_stats(train_df,
    #                   features=model.feature_names,
    #                   predictors=model.target_name,
    #                   categorical=mod_pr_pre_vol.categorical,
    #                   importance=feature_importances,
    #                   category_labels=model.category_labels_overrides)

    #remove_context()
    print("All done!")