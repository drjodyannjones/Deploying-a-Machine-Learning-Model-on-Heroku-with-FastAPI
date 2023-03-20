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
    if training:
        df_encoded = pd.get_dummies(df, columns=categorical_features)
        one_hot_columns = df_encoded.columns
    else:
        one_hot_columns = le.classes_
        df_encoded = pd.get_dummies(df, columns=categorical_features)
        for col in one_hot_columns:
            if col not in df_encoded.columns:
                df_encoded[col] = 0

        df_encoded = df_encoded[one_hot_columns]

    # If training, fit the encoders and save them, otherwise, use the provided encoders
    if training:
        le = LabelEncoder()
        lb = LabelBinarizer()
        le.fit(df[label])
        lb.fit(df[label])

    # Encode the labels
    y = lb.transform(df[label])

    # Remove the label column
    df_encoded = df_encoded.drop(label, axis=1)

    # Scale the numerical features
    scaler = StandardScaler()
    df_encoded[df_encoded.columns] = scaler.fit_transform(df_encoded[df_encoded.columns])

    return df_encoded, y, le, lb, scaler


