import sys
sys.path.append("src/app")

import joblib
from fastapi import APIRouter, Depends
import pandas as pd
from pydantic import BaseModel
from models.data import process_data
from models.train_model import inference
from utils.utils import load_model, load_encoder, load_label_binarizer

router = APIRouter()

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

class InputData(BaseModel):
    age: int
    workclass: str
    fnlwgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

@router.get("/")
def welcome():
    return {"message": "Welcome to the salary prediction API"}

@router.post("/predict")
def predict(input_data: InputData, model=Depends(load_model), encoder=Depends(load_encoder), lb=Depends(load_label_binarizer)):
    # Convert input data to a DataFrame
    data = pd.DataFrame([dict(input_data)])

    # Process the input data with the process_data function
    X, _, _, _ = process_data(
        data, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
    )

    # Run inference on the processed data
    preds = inference(model, X)

    # Convert the binary prediction to its corresponding label
    prediction = lb.inverse_transform(preds)[0]

    return {"prediction": prediction}
