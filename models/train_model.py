# Script to train machine learning model.
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from models.data import process_data
import joblib
import os

# Add the necessary imports for the starter code.
from models.model import train_model, compute_model_metrics, inference
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer

# Add code to load in the data.
data_path = 'data/census.csv'  # Replace with the path to your data file
data = pd.read_csv(data_path)

# Optional enhancement, use K-fold cross validation instead of a train-test split.
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
    train, categorical_features=cat_features, label="", training=True
)

# Save the fitted OneHotEncoder
encoder_path = 'starter/starter/encoder.pkl'  # Replace with the path where you want to save your encoder
joblib.dump(encoder, encoder_path)

# Save the fitted LabelBinarizer
lb_path = 'starter/starter/label_binarizer.pkl'  # Replace with the path where you want to save your label binarizer
joblib.dump(lb, lb_path)

# Process the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="", training=False, encoder=encoder, lb=lb
)

# Train and save a model.
model = train_model(X_train, y_train)

# Save the trained model
model_path = 'starter/starter/model.pkl'  # Replace with the path where you want to save your model
joblib.dump(model, model_path)

# Run inference on the test data
preds = inference(model, X_test)

# Compute model metrics
precision, recall, fbeta = compute_model_metrics(y_test, preds)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {fbeta}")
