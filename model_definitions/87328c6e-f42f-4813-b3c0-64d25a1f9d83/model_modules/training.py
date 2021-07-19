from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from nyoka import xgboost_to_pmml
from teradataml import create_context
from teradataml.context.context import *
from teradataml.dataframe.dataframe import DataFrame
from tdfs.featurestore import FeatureStore
from aoa.stats import stats
from aoa.util.artefacts import save_plot

import joblib
import os


def train(data_conf, model_conf, **kwargs):
    hyperparams = model_conf["hyperParameters"]

    create_context(host=os.environ["AOA_CONN_HOST"],
                   username=os.environ["AOA_CONN_USERNAME"],
                   password=os.environ["AOA_CONN_PASSWORD"],
                   database=data_conf["schema"] if "schema" in data_conf and data_conf["schema"] != "" else None)

    fs = FeatureStore(get_connection(), data_conf['feature_store'])

    entity = data_conf['entity']
    feature_names = data_conf['feature_names'].split(',')
    target_name = data_conf['target_name']
    training_dataset_instance_date = data_conf['instance_date']

    # read training dataset from Feature Store and convert to Pandas Dataframe
    train_df = fs.get_featureset_df(entity, training_dataset_instance_date, feature_names + [target_name])
    train_pdf = train_df.to_pandas()

    # split data into X and y
    X_train = train_pdf.drop(target_name, 1)
    y_train = train_pdf[target_name]

    print("Starting training...")

    # fit model to training data
    model = Pipeline([('scaler', MinMaxScaler()),
                      ('xgb', XGBClassifier(eta=hyperparams["eta"],
                                            max_depth=hyperparams["max_depth"]))])
    # xgboost saves feature names but lets store on pipeline for easy access later
    model.feature_names = feature_names
    model.target_name = target_name

    model.fit(X_train, y_train)

    print("Finished training")

    # export model artefacts
    joblib.dump(model, "artifacts/output/model.joblib")

    # we can also save as pmml so it can be used for In-Vantage scoring etc.
    xgboost_to_pmml(pipeline=model, col_names=feature_names, target_name=target_name,
                    pmml_f_name="artifacts/output/model.pmml")

    print("Saved trained model")

    from xgboost import plot_importance
    model["xgb"].get_booster().feature_names = feature_names
    plot_importance(model["xgb"].get_booster(), max_num_features=10)
    save_plot("feature_importance.png")

    feature_importance = model["xgb"].get_booster().get_score(importance_type="weight")
    stats.record_training_stats(train_df,
                       features=feature_names,
                       predictors=[target_name],
                       categorical=[target_name],
                       importance=feature_importance,
                       category_labels={target_name: {0: "false", 1: "true"}})
