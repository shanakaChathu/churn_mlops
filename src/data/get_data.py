import yaml
import argparse
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split

def read_params(config_path):
    """
    read parameters from the params.yaml file
    input: params.yaml location
    output: parameters as dictionary
    """
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

def load_data(data_path):
    """
    load csv dataset from given path 
    input: csv path 
    output:pandas dataframe 
    """
    df = pd.read_csv(data_path, sep=",", encoding='utf-8')
    return df

def get_feat_and_target(df,target):
    """
    Get features and target variables seperately from given dataframe and target 
    input: dataframe and target column
    output: two dataframes for x and y 
    """
    x=df.drop(target,axis=1)
    y=df[[target]]
    return x,y    

def external_to_raw(config_path):
    """
    load data from external location(data/external) to the raw folder(data/raw) with train and teting dataset 
    input: config_path 
    output: saving train/test_week1/test_week2 in data/raw folder 
    """
    config=read_params(config_path)
    external_data_path=config["external_data_config"]["external_data_csv"]
    target=config["external_data_config"]["target"]
    raw_train_path =config["external_data_config"]["raw_train"]
    raw_test_week1_path =config["external_data_config"]["raw_test_week1"]
    raw_test_week2_path =config["external_data_config"]["raw_test_week2"]
    train_split = config["external_data_config"]["split_ratio_train"]
    test_split = config["external_data_config"]["split_ratio_test"]

    df=load_data(external_data_path)
    x,y=get_feat_and_target(df,target)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=train_split, random_state=1,stratify=y)
    x_week1, x_week2, y_week1, y_week2 = train_test_split(x_test, y_test, test_size=test_split, random_state=1)

    train=pd.concat([x_train,y_train],axis=1)
    test_week1=pd.concat([x_week1,y_week1],axis=1)
    test_week2=pd.concat([x_week2,y_week2],axis=1)
    
    train.to_csv(raw_train_path,index=False)
    test_week1.to_csv(raw_test_week1_path,index=False)
    test_week2.to_csv(raw_test_week2_path,index=False)

    print("Train dataset size: ",train.shape)
    print("Week1 dataset size: ",test_week1.shape)
    print("Week2 dataset size: ",test_week2.shape)

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    external_to_raw(config_path=parsed_args.config)