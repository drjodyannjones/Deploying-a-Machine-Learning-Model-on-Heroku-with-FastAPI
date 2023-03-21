import joblib
from fastapi import Depends
from functools import lru_cache

model_path = 'src/app/models/model.pkl'
encoder_path = 'src/app/models/encoder.pkl'
lb_path = 'src/app/models/label_binarizer.pkl'

@lru_cache
def load_model():
    return joblib.load(model_path)

@lru_cache
def load_encoder():
    return joblib.load(encoder_path)

@lru_cache
def load_label_binarizer():
    return joblib.load(lb_path)
