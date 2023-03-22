import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

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

model_path = os.path.join(MODELS_DIR, "model.pkl")
ct_path = os.path.join(MODELS_DIR, "column_transformer.pkl")
scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
lb_path = os.path.join(MODELS_DIR, "label_binarizer.pkl")
