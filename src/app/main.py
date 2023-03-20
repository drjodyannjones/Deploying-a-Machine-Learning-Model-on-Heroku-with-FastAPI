import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from models.data import process_data
from models.train_model import inference
from config import cat_features, model_path, encoder_path, lb_path
from src.api.endpoints import router as api_router

app = FastAPI()

app.include_router(api_router)


app = FastAPI()

def initialize():
    model = joblib.load(model_path)
    encoder = joblib.load(encoder_path)
    lb = joblib.load(lb_path)
    return model, encoder, lb

model, encoder, lb = initialize()

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

@app.get("/")
def welcome():
    return {"message": "Welcome to the salary prediction API"}

@app.post("/predict")
def predict(input_data: InputData):
    # Convert input data to a DataFrame
    data = pd.DataFrame([dict(input_data)])

    # Process the input data with the process_data function
    X, _, _, _, _ = process_data(
        data, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
    )

    # Run inference on the processed data
    preds = inference(model, X)

    # Convert the binary prediction to its corresponding label
    prediction = lb.inverse_transform(preds)[0]

    return {"prediction": prediction}
