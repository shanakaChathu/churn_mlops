# split the raw data 
# save it in data/processed folder
import os
import argparse
import pandas as pd
from get_data import read_params
from sklearn.model_selection import train_test_split

def split_and_saved_data(config_path):
    config = read_params(config_path)
    test_data_path = config["raw_data_config"]["test_data_csv"] 
    train_data_path = config["raw_data_config"]["train_data_csv"]
    raw_data_path = config["raw_data_config"]["raw_data_csv"]
    split_ratio = config["raw_data_config"]["train_test_split_ratio"]
    #random_state = config["base"]["random_state"]

    df = pd.read_csv(raw_data_path, sep=",")
    train, test = train_test_split(
        df, 
        test_size=split_ratio, 
        random_state=1
        )
    train.to_csv(train_data_path, sep=",", index=False, encoding="utf-8")
    test.to_csv(test_data_path, sep=",", index=False, encoding="utf-8")

if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    split_and_saved_data(config_path=parsed_args.config)