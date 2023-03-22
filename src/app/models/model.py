import os
import numpy as np
import pandas as pd
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from src.app.models.data import process_data # Import the process_data function from data.py

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
    return preds.astype(int)

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
    X, y, encoder, lb, scaler, _ = process_data(data, cat_features, label='salary', training=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = train_model(X_train, y_train)
    preds = inference(model, X_test)
    precision, recall, f1 = compute_model_metrics(y_test, preds)

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")

    performance_on_slices(model, data.iloc[X_test.index], encoder, lb)
