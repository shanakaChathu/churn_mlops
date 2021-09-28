# Libraries 
import argparse
import joblib
import json
import mlflow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import category_encoders as ce
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from src.get_data import read_params
from urllib.parse import urlparse
#from src.get_data import get_feat_and_target
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

def accuracymeasures(y_test,predictions):
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='weighted')
    recall = recall_score(y_test, predictions, average='weighted')
    f1score = f1_score(y_test, predictions, average='weighted')
    target_names = ['0','1']
    print("Classification report")
    print("---------------------","\n")
    print(classification_report(y_test, predictions,target_names=target_names),"\n")
    print("Confusion Matrix")
    print("---------------------","\n")
    print(confusion_matrix(y_test, predictions),"\n")

    print("Accuracy Measures")
    print("---------------------","\n")
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 Score: ", f1score)

def get_feat_and_target(df,target):
    """
    Get features and target variables seperately from given dataframe and target 
    input: dataframe and target column
    output: two dataframes for x and y 
    """
    x=df.drop(target,axis=1)
    y=df[[target]]
    return x,y    

def train_and_evaluate(config_path):
    config = read_params(config_path)
    test_data_path = config["raw_data_config"]["test_data_csv"] 
    train_data_path = config["raw_data_config"]["train_data_csv"]
    target = config["raw_data_config"]["target"]
    #model_dir = config["model_dir"]

    train = pd.read_csv(train_data_path, sep=",")
    test = pd.read_csv(test_data_path, sep=",")

    num_var = [feature for feature in train.columns if train[feature].dtypes != 'O']
    num_var.append(target)
    train=train[num_var]
    test=test[num_var]

    train_x,train_y=get_feat_and_target(train,target)
    test_x,test_y=get_feat_and_target(test,target)

    rfc = RandomForestClassifier()
    rfc.fit(train_x, train_y)
    y_pred = rfc.predict(test_x)

    accuracymeasures(test_y,y_pred)
 
if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    train_and_evaluate(config_path=parsed_args.config)

