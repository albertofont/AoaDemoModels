from sklearn import metrics
from teradataml import create_context, remove_context
from teradataml.dataframe.dataframe import DataFrame
from sklearn.metrics import mean_squared_error,mean_absolute_error
import pandas as pd
import numpy as np

import os
import joblib
import json


def save_plot(title):
    import matplotlib.pyplot as plt

    plt.title(title)
    fig = plt.gcf()
    filename = title.replace(" ", "_").lower()
    fig.savefig('artifacts/output/{}'.format(filename), dpi=500)
    plt.clf()


def evaluate(data_conf, model_conf, **kwargs):
    mod_pr_pre_vol = joblib.load('artifacts/input/model.joblib')

    create_context(host=os.environ["AOA_CONN_HOST"],
                   username=os.environ["AOA_CONN_USERNAME"],
                   password=os.environ["AOA_CONN_PASSWORD"])

    # Read test dataset from Teradata
    # As this is for demo purposes, we simulate the test dataset changing between executions
    # by introducing a random sample. Note that the sampling is performed in Teradata!
    
    print("USERNAME: ",os.environ["AOA_CONN_USERNAME"])
    print("data_conf: ",data_conf)    
    print("data_conf table: ",data_conf["table"])
    test_df = DataFrame(data_conf["table"])
    test_df = test_df.to_pandas()
    remove_context()
    
    test_df.set_index("DATESTAMP", inplace=True)
    
    print(test_df.head(1))
    print(test_df.tail(1))

    # Pre-processing the Data
    # Extracting Closing Prices
    test_df['vol'] = test_df['VOW3.DE_Close']
    test_df['por'] = test_df['PAH3.DE_Close']
    test_df['bmw'] = test_df['BMW.DE_Close']
    # Creating Returns
    test_df['ret_vol'] = test_df['vol'].pct_change(1).mul(100)
    test_df['ret_por'] = test_df['por'].pct_change(1).mul(100)
    test_df['ret_bmw'] = test_df['bmw'].pct_change(1).mul(100)
    # Extracting Volume
    test_df['vol'] = test_df['VOW3.DE_Close']
    test_df['por'] = test_df['PAH3.DE_Close']
    test_df['bmw'] = test_df['BMW.DE_Close']
    
    X_test = test_df[mod_pr_pre_vol.feature_names]
    y_test = test_df[mod_pr_pre_vol.target_name]

     
    print("Scoring")
    #y_pred = mod_pr_pre_vol.predict(test_df[model.feature_names])
    
    df_auto_pred_pr = pd.DataFrame(mod_pr_pre_vol.predict(n_periods = len(test_df),exogenous = test_df[mod_pr_pre_vol.feature_names]),
                               index = test_df.index)
    

    evaluation = {
        'MAE': '{:.2f}'.format(metrics.mean_absolute_error(test_df['vol'],df_auto_pred_pr)),
        'MSE': '{:.2f}'.format(metrics.mean_squared_error(test_df['vol'],df_auto_pred_pr)),
        'RMSE': '{:.2f}'.format(np.sqrt(metrics.mean_squared_error(test_df['vol'],df_auto_pred_pr)))
        
#        'Accuracy': '{:.2f}'.format(metrics.accuracy_score(y_test, y_pred)),
#        'Recall': '{:.2f}'.format(metrics.recall_score(y_test, y_pred)),
#        'Precision': '{:.2f}'.format(metrics.precision_score(y_test, y_pred)),
#        'f1-score': '{:.2f}'.format(metrics.f1_score(y_test, y_pred))
    }

    with open("artifacts/output/metrics.json", "w+") as f:
        json.dump(evaluation, f)

    test_df['vol'].plot(figsize = (20,8),label='TEST',legend=True)
    df_auto_pred_pr.squeeze().plot(label='PRED',legend=True)
    
    #metrics.plot_confusion_matrix(model, X_test, y_test)
    save_plot('Predictions_Exog_vs_Real_Data')

#    metrics.plot_roc_curve(model, X_test, y_test)
#    save_plot('ROC Curve')

    # xgboost has its own feature importance plot support but lets use shap as explainability example
    import shap

 #   shap_explainer = shap.TreeExplainer(model['xgb'])
 #   shap_values = shap_explainer.shap_values(X_test)

 #   shap.summary_plot(shap_values, X_test, feature_names=model.feature_names,
 #                     show=False, plot_size=(12,8), plot_type='bar')
 #   save_plot('SHAP Feature Importance')
