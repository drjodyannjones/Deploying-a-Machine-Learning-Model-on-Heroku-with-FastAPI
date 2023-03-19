import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

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

def performance_on_slices(model: RandomForestClassifier, X_test: np.array, y_test: np.array, test_df: pd.DataFrame, categorical_feature: str):
    unique_categories = test_df[categorical_feature].unique()
    results = {}

    for category in unique_categories:
        mask = test_df[categorical_feature] == category
        X_test_slice = X_test[mask]
        y_test_slice = y_test[mask]

        if len(X_test_slice) > 0:
            preds = inference(model, X_test_slice)
            precision, recall, f1 = compute_model_metrics(y_test_slice, preds)
            results[category] = {"Precision": precision, "Recall": recall, "F1 Score": f1}

    return results

if __name__ == '__main__':
    data = pd.read_csv("data/census.csv", delimiter=",")
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    test_df = pd.DataFrame(X_test, columns=data.columns[:-1])
    test_df['salary'] = y_test

    model = train_model(X_train, y_train)
    preds = inference(model, X_test)
    precision, recall, f1 = compute_model_metrics(y_test, preds)

    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

    # Performance on slices of the test DataFrame
    slice_performance = performance_on_slices(model, X_test, y_test, test_df, 'education')

    for category, metrics in slice_performance.items():
        print(f"\n{category}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
