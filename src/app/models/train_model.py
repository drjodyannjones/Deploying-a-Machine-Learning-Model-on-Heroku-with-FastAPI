# Script to train machine learning model.

import pandas as pd

from joblib import dump
from sklearn.model_selection import train_test_split
from src.app.models.data import process_data

def get_train_test_data(root_path):
    """
    Get data for training and testing

    Parameters
    ----------
    root_path

    Returns
    -------
    train_df , test_df
    """
    data = pd.read_csv(f"{root_path}/data/clean/census.csv")
    train_df, test_df = train_test_split(data, test_size=0.20)

    return train_df, test_df


def train_save_model(train, cat_features, root_path):
    """
    Train and save ml model
    Parameters
    ----------
    root_path
    train
    cat_features

    Returns
    -------

    """
    x_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )
    # train model
    trained_model = train_model(x_train, y_train)
    # save model
    dump(trained_model, f"{root_path}/model/model.joblib")
    dump(encoder, f"{root_path}/model/encoder.joblib")
    dump(lb, f"{root_path}/model/lb.joblib")