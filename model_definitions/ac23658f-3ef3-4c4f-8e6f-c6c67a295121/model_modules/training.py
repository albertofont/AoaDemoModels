from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from nyoka import xgboost_to_pmml
from teradataml import create_context
from teradataml.dataframe.dataframe import DataFrame
from aoa.stats import stats
from aoa.util.artefacts import save_plot
from datetime import datetime
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pylab as plt

import joblib
import osf


def train(data_conf, model_conf, **kwargs):
    hyperparams = model_conf["hyperParameters"]

    create_context(host=os.environ["AOA_CONN_HOST"],
                   username=os.environ["AOA_CONN_USERNAME"],
                   password=os.environ["AOA_CONN_PASSWORD"],
                   database=data_conf["schema"] if "schema" in data_conf and data_conf["schema"] != "" else None)

    
    feature_names = ["NumTimesPrg", "PlGlcConc", "BloodP", "SkinThick", "TwoHourSerIns", "BMI", "DiPedFunc", "Age"]
    target_name = "HasDiabetes"

    # read training dataset from Teradata and convert to pandas
    data = DataFrame(data_conf["table"]).to_pandas()
    data = data.drop(['Unnamed: 15', 'Unnamed: 16'], axis=1)
    data = data.dropna()
    data = data.replace(-200, np.nan)
    data.loc[:,'Datetime'] = data['Date'] + ' ' + data['Time']
    data['CO(GT)'] = data['CO(GT)'].str.replace(',', '.').astype(float)
    data['C6H6(GT)'] = data['C6H6(GT)'].str.replace(',','.').astype(float)
    data['T'] = data['T'].str.replace(',', '.').astype(float)
    data['RH'] = data['RH'].str.replace(',', '.').astype(float)
    data['AH'] = data['AH'].str.replace(',', '.').astype(float)
    data = data.replace(-200, np.nan)

    DateTime = []
    for x in data['Datetime']:
        DateTime.append(datetime.strptime(x,'%d/%m/%Y %H.%M.%S'))
    datetime_ = pd.Series(DateTime)
    data.index = datetime_
        
    S1 = data['PT08.S1(CO)'].fillna(data['PT08.S1(CO)'].mean())
    S2 = data['PT08.S2(NMHC)'].fillna(data['PT08.S1(CO)'].mean())
    S3 = data['PT08.S3(NOx)'].fillna(data['PT08.S1(CO)'].mean())
    S4 = data['PT08.S4(NO2)'].fillna(data['PT08.S1(CO)'].mean())
    S5 = data['PT08.S5(O3)'].fillna(data['PT08.S1(CO)'].mean())
    S6 = data['RH'].fillna(data['PT08.S1(CO)'].mean())
    df_RH = pd.DataFrame({'S6':S6})
    df_RH_logScale = np.log(df_RH)
    
    #train_df = train_df.select([feature_names + [target_name]])
    #train_pdf = train_df.to_pandas()

    # split data into X and y
    #X_train = train_pdf.drop(target_name, 1)
    #y_train = train_pdf[target_name]

    print("Starting training...")
    
    # fit model to training data
    #model = Pipeline([('scaler', MinMaxScaler()),
    #                  ('xgb', XGBClassifier(eta=hyperparams["eta"],
    #                                        max_depth=hyperparams["max_depth"]))])
    # xgboost saves feature names but lets store on pipeline for easy access later
    #model.feature_names = feature_names
    #model.target_name = target_name

    #model.fit(X_train, y_train)

    model = ARIMA(df_RH_logScale,order=(2, 1, 2))

    result_AR= model.fit()
    
    print("Finished training")

    # export model artefacts
    joblib.dump(result_AR, "artifacts/output/model.joblib")

    # we can also save as pmml so it can be used for In-Vantage scoring etc.
    # xgboost_to_pmml(pipeline=model, col_names=feature_names, target_name=target_name,
    #                pmml_f_name="artifacts/output/model.pmml")

    print("Saved trained model")

#    from xgboost import plot_importance
#    model["xgb"].get_booster().feature_names = feature_names
#    plot_importance(model["xgb"].get_booster(), max_num_features=10)
#    save_plot("feature_importance.png")

#    feature_importance = model["xgb"].get_booster().get_score(importance_type="weight")
#    stats.record_training_stats(train_df,
#                       features=feature_names, # lista
#                       predictors=[target_name], # listade 1 elemento
#                       categorical=[target_name], #. lista de 
#                       importance=feature_importance, #dict vacío
#                       category_labels={target_name: {0: "false", 1: "true"}}) # vacio

    datasetLogDiffShifting  = df_RH_logScale-df_RH_logScale.shift()
    datasetLogDiffShifting.dropna(inplace=True)
    plt.plot(datasetLogDiffShifting)
    plt.plot(df_RH_logScale,color='orange')
    plt.plot(result_AR.fittedvalues,color='red')
    plt.title('RSS: %.4f'%sum((result_AR.fittedvalues-datasetLogDiffShifting["S6"])**2))
    save_plot("feature_importance.png")
    print('Plotting AR Model')

