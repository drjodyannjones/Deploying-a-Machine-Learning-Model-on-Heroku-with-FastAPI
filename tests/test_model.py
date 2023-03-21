import numpy as np
import pandas as pd
from src.app.models.model import train_model, compute_model_metrics, inference

def test_train_model():
    X_train = np.array([[1, 2], [3, 4]])
    y_train = np.array([0, 1])

    model = train_model(X_train, y_train)
    assert model is not None, "Model not returned"

def test_compute_model_metrics():
    y = np.array([1, 0, 1, 0])
    preds = np.array([1, 0, 0, 1])

    precision, recall, fbeta = compute_model_metrics(y, preds)

    assert precision == 0.5, "Incorrect precision"
    assert recall == 0.5, "Incorrect recall"
    assert fbeta == 0.5, "Incorrect F1 score"

def test_inference():
    X = np.array([[1, 2], [3, 4]])
    model = train_model(X, np.array([0, 1]))
    preds = inference(model, X)

    assert len(preds) == 2, "Incorrect number of predictions"
    assert all(isinstance(p, int) for p in preds), "Predictions should be integers"
