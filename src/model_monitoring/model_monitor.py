import json
import joblib
import mlflow
import argparse
import numpy as np
import pandas as pd
from datetime import date
from dateutil import parser
from src.get_data import read_params
from sklearn import datasets, model_selection, neighbors,linear_model
from evidently.dashboard import Dashboard
from evidently.tabs import ClassificationPerformanceTab, DataDriftTab,CatTargetDriftTab,ProbClassificationPerformanceTab
from evidently.model_profile import Profile
from evidently.profile_sections import ClassificationPerformanceProfileSection,DataDriftProfileSection,CatTargetDriftProfileSection,base_profile_section,ProbClassificationPerformanceProfileSection


def model_monitoring(config_path):
    config = read_params(config_path)
    test_data_path=config["external_data_config"]["raw_test_week1"]
    train_data_path = config["raw_data_config"]["train_data_csv"]
    target = config["raw_data_config"]["target"]
    monitor_dashboard_path = config["model_monitor"]["monitor_dashboard_html"]
    monitor_target = config["model_monitor"]["target_col_name"]

    ref=pd.read_csv(train_data_path)
    cur=pd.read_csv(test_data_path)

    num_var = [feature for feature in ref.columns if ref[feature].dtypes != 'O']
    num_var.append(target)
    ref=ref[num_var]
    cur=cur[num_var]

    ref=ref.rename(columns ={'churn':monitor_target}, inplace = False)
    cur=cur.rename(columns ={'churn':monitor_target}, inplace = False)
    
    data_and_target_drift_dashboard = Dashboard(tabs=[DataDriftTab, CatTargetDriftTab])
    data_and_target_drift_dashboard.calculate(ref,cur, column_mapping = None)
    data_and_target_drift_dashboard.save(monitor_dashboard_path)

if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    model_monitoring(config_path=parsed_args.config)