import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, LabelBinarizer, StandardScaler

def process_data(df: pd.DataFrame, categorical_features: list, label: str, training: bool=True, le=None, lb=None):
    """
    Preprocess data.

    Inputs
    ------
    df : pandas.DataFrame
        Data to be processed.
    categorical_features : list
        Names of the categorical features.
    label : str
        Name of the target variable.
    training : bool
        Whether the function is called for training or testing.
        When True, a new LabelEncoder and LabelBinarizer are fit on the data.
        When False, they are assumed to be fitted previously.
    le : sklearn.preprocessing.LabelEncoder, optional
        Pre-fitted LabelEncoder object.
    lb : sklearn.preprocessing.LabelBinarizer, optional
        Pre-fitted LabelBinarizer object.

    Returns
    -------
    X : numpy.ndarray
        Processed data.
    y : numpy.ndarray
        Target variable.
    le : sklearn.preprocessing.LabelEncoder or None
        LabelEncoder object fitted on the target variable.
    lb : sklearn.preprocessing.LabelBinarizer or None
        LabelBinarizer object fitted on the target variable.
    scaler : sklearn.preprocessing.StandardScaler
        StandardScaler object fitted on the processed data.
    """
    print("Columns in DataFrame:")
    print(df.columns)

    X = df.drop(label, axis=1)
    y = df[label]

    if training:
        le = LabelEncoder()
        y = le.fit_transform(y)

        lb = LabelBinarizer()
        y = lb.fit_transform(y)

    else:
        assert le is not None, "LabelEncoder not provided for testing data."
        assert lb is not None, "LabelBinarizer not provided for testing data."

        y = le.transform(y)
        y = lb.transform(y)

    # Convert categorical variables to one-hot encoded variables
    for feature in categorical_features:
        X[feature] = X[feature].astype("category")

    X = pd.get_dummies(X, columns=categorical_features)

    # Scale numerical features
    scaler = StandardScaler()
    numerical_features = [
        col for col in X.columns if col not in categorical_features
    ]
    X[numerical_features] = scaler.fit_transform(X[numerical_features])

    return X.to_numpy(), y, le, lb, scaler
