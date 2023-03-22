import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, LabelBinarizer, StandardScaler

from typing import List, Tuple, Optional, Union

def process_data(
    df: pd.DataFrame,
    categorical_features: List[str],
    label: str,
    training: bool = True,
    encoder: Optional[LabelEncoder] = None,
    lb: Optional[LabelBinarizer] = None,
    scaler: Optional[StandardScaler] = None
) -> Tuple[pd.DataFrame, np.array, LabelEncoder, LabelBinarizer, StandardScaler, List[str]]: # type: ignore
    """
    Process the input dataset, applying one-hot encoding and scaling.

    Args:
        df (pd.DataFrame): The input data.
        categorical_features (list): The categorical feature columns.
        label (str): The target column.
        training (bool): Whether the data is for training or testing.
        encoder (LabelEncoder): The pre-fit label encoder (for testing).
        lb (LabelBinarizer): The pre-fit label binarizer (for testing).
        scaler (StandardScaler): The pre-fit scaler (for testing).

    Returns:
        pd.DataFrame: The processed data.
        np.array: The target labels.
        LabelEncoder: The label encoder.
        LabelBinarizer: The label binarizer.
        StandardScaler: The scaler.
        List[str]: The column names after one-hot encoding.
    """
    # One-hot encode the categorical features
    if training:
        df_encoded = pd.get_dummies(df, columns=categorical_features)
        one_hot_columns = df_encoded.columns
    else:
        one_hot_columns = encoder.classes_
        df_encoded = pd.get_dummies(df, columns=categorical_features)
        for col in one_hot_columns:
            if col not in df_encoded.columns:
                df_encoded[col] = 0

        df_encoded = df_encoded[one_hot_columns]

    # If training, fit the encoders and save them, otherwise, use the provided encoders
    if training:
        encoder = LabelEncoder()
        lb = LabelBinarizer()
        encoder.fit(df[label])
        lb.fit(df[label])

    # Encode the labels
    y = lb.transform(df[label])

    # Remove the label column
    df_encoded = df_encoded.drop(label, axis=1)

    # Scale the numerical features
    if training:
        scaler = StandardScaler()
        df_encoded[df_encoded.columns] = scaler.fit_transform(df_encoded[df_encoded.columns])
    else:
        df_encoded[df_encoded.columns] = scaler.transform(df_encoded[df_encoded.columns])

    return df_encoded, y, encoder, lb, scaler, one_hot_columns # type: ignore

