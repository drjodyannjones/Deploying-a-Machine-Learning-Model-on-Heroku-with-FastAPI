import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
import joblib
import sys

sys.path.append('src')

from src.app.models.train_model import train_model, inference, compute_model_metrics, CustomTransformer, ColumnTransformer, LabelEncoder

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

# Perform inference
preds = inference(model, X_test)

# Compute model metrics
precision, recall, fbeta = compute_model_metrics(y_test, preds)

# Test assertions
assert 0 <= precision <= 1, "Precision should be between 0 and 1"
assert 0 <= recall <= 1, "Recall should be between 0 and 1"
assert 0 <= fbeta <= 1, "F1 Score should be between 0 and 1"

print("All tests passed!")
