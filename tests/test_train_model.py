import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from src.app.models.train_model import process_data
from models.train_model import train_model, compute_model_metrics, inference
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer

def test_compute_model_metrics():
    data = pd.read_csv("data/census.csv")
    train, test = train_test_split(data, test_size=0.20)

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

    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )
    X_test, y_test, _, _ = process_data(
        test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
    )
    model = train_model(X_train, y_train)
    preds = inference(model, X_test)

    precision, recall, fbeta = compute_model_metrics(y_test, preds)

    assert precision is not None, "Precision not returned"
    assert recall is not None, "Recall not returned"
    assert fbeta is not None, "Fbeta not returned"

    # Check the type of the computed metrics
    assert isinstance(precision, float), "Precision is not a float"
    assert isinstance(recall, float), "Recall is not a float"
    assert isinstance(fbeta, float), "Fbeta is not a float"

    # Check the range of the computed metrics
    assert precision >= 0 and precision <= 1, "Precision is not within [0, 1]"
    assert recall >= 0 and recall <= 1, "Recall is not within [0, 1]"
    assert fbeta >= 0 and fbeta <= 1, "Fbeta is not within [0, 1]"
