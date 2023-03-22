import pandas as pd
import numpy as np
from src.app.models.data import process_data
from app.main import scaler



def test_process_data():
    # Load sample data for testing
    data = pd.DataFrame({
    "age": [40, 25],
    "workclass": ["Private", "Self-emp-not-inc"],
    "fnlgt": [121772, 226802],
    "education": ["Assoc-voc", "11th"],
    "education_num": [11, 7],
    "marital_status": ["Married-civ-spouse", "Never-married"],
    "occupation": ["Craft-repair", "Machine-op-inspct"],
    "relationship": ["Husband", "Own-child"],
    "race": ["Asian-Pac-Islander", "Black"],
    "sex": ["Male", "Female"],
    "capital_gain": [7298, 0],
    "capital_loss": [0, 0],
    "hours_per_week": [40, 40],
    "native_country": ["United-States", "United-States"],
    "salary": [">50K", "<=50K"]
})



    # Define categorical features
    cat_features = [
        "workclass",
        "education",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native_country",
    ]

    X_train, y_train, encoder, lb, scaler, output_columns = process_data(
        data, categorical_features=cat_features, label="salary", training=True)

    y_train = y_train.ravel()  # Add this line to make y_train 1-dimensional

    assert X_train.shape == (2, 17), "Processed data shape mismatch"
    assert y_train.shape == (1,), "Labels shape mismatch"
    assert encoder is not None, "Encoder not returned"
    assert lb is not None, "Label Binarizer not returned"
    assert output_columns is not None, "Output columns not returned"
