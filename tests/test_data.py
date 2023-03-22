import pandas as pd
import numpy as np
from src.app.models.data import process_data


def test_process_data():
    # Load sample data for testing
    data = pd.DataFrame({
        "age": 40,
        "workclass": "Private",
        "fnlwgt": 121772,
        "education": "Assoc-voc",
        "education_num": 11,
        "marital-status": "Married-civ-spouse",
        "occupation": "Craft-repair",
        "relationship": "Husband",
        "race": "Asian-Pac-Islander",
        "sex": "Male",
        "capital_gain": 7298,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native-country": "United-States"
    })

    # Define categorical features
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    X_train, y_train, encoder, lb, scaler, output_columns = process_data(
        data, categorical_features=cat_features, label="salary", training=True)

    y_train = y_train.ravel()  # Add this line to make y_train 1-dimensional

    assert X_train.shape == (2, 17), "Processed data shape mismatch"
    assert y_train.shape == (2,), "Labels shape mismatch"
    assert encoder is not None, "Encoder not returned"
    assert lb is not None, "Label Binarizer not returned"
    assert output_columns is not None, "Output columns not returned"
