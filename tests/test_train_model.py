import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
import joblib
import sys
import os

# Update the sys.path to include the 'src' directory
sys.path.append('src')

# Import necessary functions from the main script
from test_train_model import train_model, inference, compute_model_metrics, process_data

# Load the data
data_path = 'data/census.csv'
data = pd.read_csv(data_path)

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

train, test = train_test_split(data, test_size=0.20)

# Process the data
X_train, y_train, ct, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False, ct=ct, lb=lb
)

# Train the model
model = train_model(X_train, y_train)

# Perform inference
preds = inference(model, X_test)

# Compute model metrics
precision, recall, fbeta = compute_model_metrics(y_test, preds)

# Test assertions
assert 0 <= precision <= 1, "Precision should be between 0 and 1"
assert 0 <= recall <= 1, "Recall should be between 0 and 1"
assert 0 <= fbeta <= 1, "F1 Score should be between 0 and 1"

print("All tests passed!")
