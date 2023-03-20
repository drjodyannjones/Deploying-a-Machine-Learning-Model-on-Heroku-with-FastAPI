# Script to train machine learning model.
import sys
sys.path.append('src')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from app.models.data import process_data
import joblib
import os

from sklearn.metrics import precision_recall_fscore_support

def train_model(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

def inference(model, X):
    preds = model.predict(X)
    return preds

def compute_model_metrics(y_true, y_pred):
    precision, recall, fbeta, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    return precision, recall, fbeta

# Add code to load in the data.
data_path = 'data/census.csv'  # Replace with the path to your data file
data = pd.read_csv(data_path)

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

encoder_path = 'src/app/models/encoder.pkl'
joblib.dump(encoder, encoder_path)

lb_path = 'src/app/models/label_binarizer.pkl'
joblib.dump(lb, lb_path)

X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)

model = train_model(X_train, y_train)

model_path = 'src/app/models/model.pkl'  # Updated model path
joblib.dump(model, model_path)

preds = inference(model, X_test)

precision, recall, fbeta = compute_model_metrics(y_test, preds)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {fbeta}")
