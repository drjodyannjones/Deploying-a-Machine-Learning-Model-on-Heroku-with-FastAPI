import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, LabelBinarizer, StandardScaler, OneHotEncoder

from typing import List, Tuple, Optional, Union

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, LabelBinarizer, StandardScaler, OneHotEncoder

from typing import List, Tuple, Optional, Union

def process_data(
    df: pd.DataFrame,
    categorical_features: List[str],
    label: Optional[str] = 'salary',
    training: bool = True,
    encoder: Optional[OneHotEncoder] = None,
    lb: Optional[LabelBinarizer] = None,
    scaler: Optional[StandardScaler] = None  # Add the scaler argument
) -> Tuple[pd.DataFrame, pd.Series, OneHotEncoder, LabelBinarizer, StandardScaler]:  # Update the return type hint

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
        scaler = StandardScaler()  # Initialize the scaler
        encoder.fit(df[label])
        lb.fit(df[label])

    # Encode the labels
    y = lb.transform(df[label])

    # Remove the label column
    df_encoded = df_encoded.drop(label, axis=1)

    # Scale the numerical features
    if training:
        df_encoded[df_encoded.columns] = scaler.fit_transform(df_encoded[df_encoded.columns])
    else:
        df_encoded[df_encoded.columns] = scaler.transform(df_encoded[df_encoded.columns])

    return df_encoded, y, encoder, lb, scaler  # Update the return statement


