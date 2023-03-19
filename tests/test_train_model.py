import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from starter.ml.data import process_data
from starter.ml.model import train_model, compute_model_metrics, inference
from sklearn.preprocessing import LabelBinarizer, LabelEncoder


def test_train_model():
    X_train = pd.DataFrame({
        "age": [25, 45],
        "workclass": ["Private", "Self-emp-not-inc"],
        "education": ["Bachelors", "HS-grad"],
        "marital-status": ["Never-married", "Married-civ-spouse"],
        "occupation": ["Adm-clerical", "Exec-managerial"],
        "relationship": ["Not-in-family", "Husband"],
        "race": ["White", "Black"],
        "sex": ["Male", "Female"],
        "native-country": ["United-States", "Mexico"],
    })
    y_train = pd.Series([">50K", "<=50K"])

    model = train_model(X_train, y_train)
    assert model is not None, "Model not returned"


def test_compute_model_metrics():
    y = pd.Series([1, 0, 1, 0])
    preds = pd.Series([1, 0, 0, 1])

    precision, recall, fbeta = compute_model_metrics(y, preds)

    assert precision == 0.5, "Incorrect precision"
    assert recall == 0.5, "Incorrect recall"
    assert fbeta == 0.5, "Incorrect F1 score"


def test_inference():
    X = pd.DataFrame({
        "age": [25, 45],
        "workclass": ["Private", "Self-emp-not-inc"],
        "education": ["Bachelors", "HS-grad"],
        "marital-status": ["Never-married", "Married-civ-spouse"],
        "occupation": ["Adm-clerical", "Exec-managerial"],
        "relationship": ["Not-in-family", "Husband"],
        "race": ["White", "Black"],
        "sex": ["Male", "Female"],
        "native-country": ["United-States", "Mexico"],
    })
    y = pd.Series([">50K", "<=50K"])
    model = train_model(X, y)
    preds = inference(model, X)

    assert len(preds) == 2, "Incorrect number of predictions"
    assert all(isinstance(p, int) for p in preds), "Predictions should be integers"


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

    X_train, y_train, encoder, lb, scaler = process_data(
        data, categorical_features
