import os
import sys
import json
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient

# Add the path to the src folder to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from app.main import app
from app.main import scaler, encoder, lb, model_columns

client = TestClient(app)


def test_welcome() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the salary prediction API"}


def test_predict_above_50k() -> None:
    input_data = {
        "age": 40,
        "workclass": "Private",
        "fnlgt": 121772,
        "education": "Assoc-voc",
        "education_num": 11,
        "marital_status": "Married-civ-spouse",
        "occupation": "Craft-repair",
        "relationship": "Husband",
        "race": "Asian-Pac-Islander",
        "sex": "Male",
        "capital_gain": 7298,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States"
    }

    # Convert input data to DataFrame
    input_df = pd.DataFrame.from_dict([input_data])

    # Select only relevant columns
    input_df = input_df[model_columns]

    # Encode categorical variables
    encoded_data = encoder.transform(input_df)

    # Scale numerical variables
    scaled_data = scaler.transform(encoded_data)

    # Make prediction
    response = client.post("/predict", json=json.loads(scaled_data.to_json(orient='values')))
    assert response.status_code == 200
    assert response.json() == {"prediction": ">50K"}


def test_predict_below_50k() -> None:
    input_data = {
        "age": 25,
        "workclass": "Private",
        "fnlgt": 226802,
        "education": "11th",
        "education_num": 7,
        "marital_status": "Never-married",
        "occupation": "Machine-op-inspct",
        "relationship": "Own-child",
        "race": "Black",
        "sex": "Male",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States"
    }

    # Convert input data to DataFrame
    input_df = pd.DataFrame.from_dict([input_data])

    # Select only relevant columns
    input_df = input_df[model_columns]

    # Encode categorical variables
    encoded_data = encoder.transform(input_df)

    # Scale numerical variables
    scaled_data = scaler.transform(encoded_data)

    # Make prediction
    response = client.post("/predict", json=json.loads(scaled_data.to_json(orient='values')))
    assert response.status_code == 200
    assert response.json() == {"prediction": "<=50K"}
