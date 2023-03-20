import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, LabelBinarizer, StandardScaler

def process_data(df, categorical_features, label, training=True, le=None, lb=None):
    """
    Process the input dataset, applying one-hot encoding and scaling.

    Args:
        df (pd.DataFrame): The input data.
        categorical_features (list): The categorical feature columns.
        label (str): The target column.
        training (bool): Whether the data is for training or testing.
        le (LabelEncoder): The pre-fit label encoder (for testing).
        lb (LabelBinarizer): The pre-fit label binarizer (for testing).

    Returns:
        pd.DataFrame: The processed data.
    """
    # One-hot encode the categorical features
    df = pd.get_dummies(df, columns=categorical_features)

    # If training, fit the encoders and save them, otherwise, use the provided encoders
    if training:
        le = LabelEncoder()
        lb = LabelBinarizer()
        le.fit(df[label])
        lb.fit(df[label])
    else:
        # Ensure that the test dataset has the same number of columns as the training dataset
        for col in le.classes_:
            if col not in df.columns:
                df[col] = 0

    # Encode the labels
    y = lb.transform(df[label])

    # Remove the label column
    df = df.drop(label, axis=1)

    # Scale the numerical features
    scaler = StandardScaler()
    df[df.columns] = scaler.fit_transform(df[df.columns])

    return df, y, le, lb, scaler

