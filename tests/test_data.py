import pandas as pd
import numpy as np
from src.app.models.data import process_data


def test_process_data():
    # Load sample data for testing
    data = pd.DataFrame({
        "age": [25, 45],
        "workclass": ["Private", "Self-emp-not-inc"],
        "education": ["Bachelors", "HS-grad"],
        "marital-status": ["Never-married", "Married-civ-spouse"],
        "occupation": ["Adm-clerical", "Exec-managerial"],
        "relationship": ["Not-in-family", "Husband"],
        "race": ["White", "Black"],
        "sex": ["Male", "Female"],
        "native-country": ["United-States", "Mexico"],
        "salary": [">50K", "<=50K"]
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
        data, categorical_features=cat_features, label="salary", training=True
    )

    assert X_train.shape == (2, 16), "Processed data shape mismatch"
    assert y_train.shape == (2,), "Labels shape mismatch"
    assert encoder is not None, "Encoder not returned"
    assert lb is not None, "Label Binarizer not returned"
    assert output_columns is not None, "Output columns not returned"
