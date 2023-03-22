import sys
sys.path.append('src')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
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


def compute_slice_performance(data, feature, model):
    unique_values = data[feature].unique()
    metrics = []
    for value in unique_values:
        df_slice = data[data[feature] == value]
        X_slice = ct.transform(df_slice.drop('salary', axis=1))
        y_slice = df_slice['salary'].values
        preds = inference(model, X_slice)
        precision, recall, fbeta = compute_model_metrics(y_slice, preds)
        metrics.append((feature, value, precision, recall, fbeta))
    return metrics


# Add code to load in the data.
data_path = 'data/census.csv'  # Replace with the path to your data file
data = pd.read_csv(data_path)

# Define cat_features list
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

train, test = train_test_split(data, test_size=0.20)

# Encode the target variable
lb = LabelEncoder()
train['salary'] = lb.fit_transform(train['salary'])
test['salary'] = lb.transform(test['salary'])

# Define column transformer

num_features = ["age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"]

ct = ColumnTransformer(
    transformers=[
        ('cat', CustomTransformer(), cat_features),
        ('num', StandardScaler(), num_features)
    ],
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

# Compute slice performance for all categorical features
all_slice_performance = []
for feature in cat_features:
    feature_slice_performance = compute_slice_performance(test, feature, model)
    all_slice_performance.extend(feature_slice_performance)

# Save the slice performance output to a file
with open('slice_output.txt', 'w') as f:
    for metric in all_slice_performance:
        f.write(f"{metric}\n")


# Save the column transformer and label encoder
ct_path = 'src/app/models/column_transformer.pkl'
joblib.dump(ct, ct_path)

scaler = ct.named_transformers_['num']  # Get the scaler from the ColumnTransformer
scaler_path = 'src/app/models/scaler.pkl'  # Define the path for the scaler
joblib.dump(scaler, scaler_path)  # Save the scaler to a pickle file

lb_path = 'src/app/models/label_binarizer.pkl'
joblib.dump(lb, lb_path)

# Save the trained model
model_path = 'src/app/models/model.pkl'
