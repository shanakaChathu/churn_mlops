import json
import joblib
import mlflow
import argparse
import numpy as np
import pandas as pd
from datetime import date
from dateutil import parser
from src.get_data import read_params

def get_feat_and_target(df,target):
    """
    Get features and target variables seperately from given dataframe and target 
    input: dataframe and target column
    output: two dataframes for x and y 
    """
    x=df.drop(target,axis=1)
    y=df[[target]]
    return x,y    

def prediction_model(config_path):
    today = date.today()
    date_key=today.strftime('%Y%m%d')

    config = read_params(config_path)
    test_data_path=config["external_data_config"]["raw_test_week1"]
    prediction_data_path= config["prediction_data_config"]["prediction_data_csv"]
    model_dir_path = config["model_dir"]

    test=pd.read_csv(test_data_path)
    num_var = [feature for feature in test.columns if test[feature].dtypes != 'O']
    test_x=test[num_var]
    
    model = joblib.load(model_dir_path)
    y_pred = model.predict(test_x)
    test['prediction']=y_pred
    test.to_csv(prediction_data_path,sep=",", index=False, encoding="utf-8")

if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    prediction_model(config_path=parsed_args.config)

