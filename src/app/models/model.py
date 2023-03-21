import os
import numpy as np
import pandas as pd
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

# Add imports and define missing variables
from sklearn.preprocessing import LabelEncoder
from typing import List

root_path = os.getcwd()
cat_features = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']

def train_model(X_train: np.array, y_train: np.array) -> RandomForestClassifier:
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def compute_model_metrics(y: np.array, preds: np.array) -> tuple:
    f1 = f1_score(y, preds, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, f1

def inference(model: RandomForestClassifier, X: np.array) -> np.array:
    preds = model.predict(X)
    return preds

# Define the missing process_data function
def process_data(data: pd.DataFrame, categorical_features: List[str], training: bool, label: str, encoder=None, lb=None):
    if training:
        encoder = LabelEncoder()
        lb = LabelEncoder()
        data[label] = lb.fit_transform(data[label])

    for col in categorical_features:
        if training:
            data[col] = encoder.fit_transform(data[col])
        else:
            data[col] = encoder.transform(data[col])

    X = data.drop(label, axis=1).values
    y = data[label].values if training else None

    return X, y, encoder, lb

def performance_on_slices(trained_model, test, encoder, lb):
    with open(f'{root_path}/slice_output.txt', 'w') as file:
        for category in cat_features:
            for cls in test[category].unique():
                temp_df = test[test[category] == cls]

                x_test, y_test, _, _ = process_data(
                    temp_df,
                    categorical_features=cat_features, training=False,
                    label="salary", encoder=encoder, lb=lb)

                y_pred = trained_model.predict(x_test)

                prc, rcl, fb = compute_model_metrics(y_test, y_pred)

                metric_info = "[%s]-[%s] Precision: %s " \
                              "Recall: %s FBeta: %s" % (category, cls,
                                                        prc, rcl, fb)
                logging.info(metric_info)
                file.write(metric_info + '\n')

if __name__ == '__main__':
    data = pd.read_csv("data/census.csv", delimiter=",")
    X, y, encoder, lb = process_data(data, cat_features, training=True, label='salary')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    test_df = pd.DataFrame(X_test, columns=data.columns[:-1])
    test_df['salary'] = y_test

    model = train_model(X_train, y_train)
    preds = inference(model, X_test)
    precision, recall, f1 = compute_model_metrics(y_test, preds)

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")
