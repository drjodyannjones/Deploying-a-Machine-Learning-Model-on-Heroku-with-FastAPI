import joblib
from fastapi import Depends
from functools import lru_cache

model_path = 'models/model.pkl'
encoder_path = 'models/encoder.pkl'
lb_path = 'models/label_binarizer.pkl'

@lru_cache
def load_model():
    return joblib.load(model_path)

@lru_cache
def load_encoder():
    return joblib.load(encoder_path)

@lru_cache
def load_label_binarizer():
    return joblib.load(lb_path)
