import sys
sys.path.append('src')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import precision_recall_fscore_support
import joblib
import os
from typing import List


class CustomTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        self.encoder = OneHotEncoder(handle_unknown='ignore')

    def fit(self, X, y=None):
        self.imputer.fit(X)
        X_imputed = self.imputer.transform(X)
        self.encoder.fit(X_imputed)
        return self

    def transform(self, X, y=None):
        X_imputed = self.imputer.transform(X)
        X_encoded = self.encoder.transform(X_imputed)
        return X_encoded

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

# Define cat_features list
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

# Encode the target variable
lb = LabelEncoder()
train['salary'] = lb.fit_transform(train['salary'])
test['salary'] = lb.transform(test['salary'])

# Define column transformer
ct = ColumnTransformer(
    transformers=[('cat', CustomTransformer(), cat_features)],
    remainder='passthrough'
)

# Fit and transform the training data
X_train = ct.fit_transform(train.drop('salary', axis=1))
y_train = train['salary'].values

# Transform the test data
X_test = ct.transform(test.drop('salary', axis=1))
y_test = test['salary'].values

# Train the model
model = train_model(X_train, y_train)

# Save the column transformer and label encoder
ct_path = 'src/app/models/column_transformer.pkl'
joblib.dump(ct, ct_path)

lb_path = 'src/app/models/label_binarizer.pkl'
joblib.dump(lb, lb_path)

# Save the trained model
model_path = 'src/app/models/model.pkl'  # Updated model path
joblib.dump(model, model_path)

# Inference
preds = inference(model, X_test)

# Compute model metrics
precision, recall, fbeta = compute_model_metrics(y_test, preds)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {fbeta}")
