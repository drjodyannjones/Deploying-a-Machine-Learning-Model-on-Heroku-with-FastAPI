import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

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

model_path = os.path.join(BASE_DIR, "models", "model.pkl")
encoder_path = os.path.join(MODELS_DIR, "encoder.pkl")
lb_path = os.path.join(MODELS_DIR, "label_binarizer.pkl")
