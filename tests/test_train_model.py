import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from models.train_model import process_data
from models.train_model import train_model, compute_model_metrics, inference
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer

def test_process_data():
    # Load sample data for testing
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
    assert X_train is not None, "X_train not returned"
    assert y_train is not None, "y_train not returned"
    assert encoder is not None, "encoder not returned"
    assert lb is not None, "lb not returned"

def test_train_model():
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
    model = train_model(X_train, y_train)
    assert model is not None, "Model not returned"

def test_inference():
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

    assert preds is not None, "Predictions not returned"
    assert len(preds) == len(y_test), "Incorrect number of predictions"

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
