import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder


def clean_dataset(df):
    """
    Clean the dataset
    """
    print(df.columns)
    df.replace({'?': None}, inplace=True)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df.drop("education-num", axis=1, inplace=True)
    df.drop("capital-gain", axis=1, inplace=True)
    df.drop("capital-loss", axis=1, inplace=True)
    return df


def data_cleaning_stage(root_path):
    """
    data cleaning
    """

    df = pd.read_csv(f"{root_path}/data/raw/census.csv", skipinitialspace=True)
    df = clean_dataset(df)
    df.to_csv(f"{root_path}/data/clean/census.csv", index=False)


def process_data(
        x,
        categorical_features=[],
        label=None,
        training=True,
        encoder=None,
        lb=None):
    """ Process the data used in the machine learning pipeline.

    Processes the data using one hot encoding for the categorical features
    and amlabel binarizer for the labels. This can be used in either
    training or inference/validation.

    Note: depending on the type of model used, you may want to add in
    functionality that scales the continuous data.

    Inputs
    ------
    X : pd.DataFrame
        Dataframe containing the features and label. Columns in
        `categorical_features`
    categorical_features: list[str]
        List containing the names of the categorical features (default=[])
    label : str
        Name of the label column in `X`. If None, then an empty array will
        be returned for y (default=None)
    training : bool
        Indicator if training mode or inference/validation mode.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.

    Returns
    -------
    X : np.array
        Processed data.
    y : np.array
        Processed labels if labeled=True, otherwise empty np.array.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained OneHotEncoder if training is True, otherwise returns the
        encoder passed in.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained LabelBinarizer if training is True, otherwise returns the
        binarizer passed in.
    """

    if categorical_features is None:
        categorical_features = list()
    if label is not None:
        y = x[label]
        x = x.drop([label], axis=1)
    else:
        y = np.array([])

    x_categorical = x[categorical_features].values
    x_continuous = x.drop(*[categorical_features], axis=1)

    if training is True:
        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        lb = LabelBinarizer()
        x_categorical = encoder.fit_transform(x_categorical)
        y = lb.fit_transform(y.values).ravel()
    else:
        x_categorical = encoder.transform(x_categorical)
        try:
            y = lb.transform(y.values).ravel()
        # Catch the case where y is None because we're doing inference.
        except AttributeError:
            pass

    x = np.concatenate([x_continuous, x_categorical], axis=1)
    return x, y, encoder, lb