import sys
sys.path.append("src/app")

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from models.data import process_data
from models.train_model import inference
from api.endpoints import router as api_router
from config import cat_features, model_path, ct_path, lb_path, scaler_path, model_columns_path


app = FastAPI()

app.include_router(api_router)

def initialize():
    model = joblib.load(model_path)
    ct = joblib.load(ct_path)  # Load the column transformer
    encoder = ct.named_transformers_['cat'].encoder  # Get the OneHotEncoder from the column transformer
    lb = joblib.load(lb_path)
    scaler = joblib.load(scaler_path)
    model_columns = joblib.load(model_columns_path)
    return model, encoder, lb, scaler, model_columns


model, encoder, lb, scaler, model_columns = initialize()

class InputData(BaseModel):
    age: int
    workclass: str
    fnlgt: int
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

    class Config:
        schema_extra = {
            "example": {
                "age": 39,
                "workclass": "State-gov",
                "fnlgt": 77516,
                "education": "Bachelors",
                "education_num": 13,
                "marital_status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "capital_gain": 2174,
                "capital_loss": 0,
                "hours_per_week": 40,
                "native_country": "United-States",
            }
        }

@app.get("/")
def welcome():
    return {"message": "Welcome to the salary prediction API"}

@app.post("/predict")
def predict(input_data: InputData):
    # Convert input data to a DataFrame
    data = pd.DataFrame([dict(input_data)])

    # Process the input data with the process_data function
    X, _, _, _, _, _ = process_data(
        data,
        categorical_features=cat_features,
        label=None,  # Pass None as label
        training=False,
        encoder=encoder,
        lb=lb,  # Pass the pre-trained lb
        scaler=scaler
    )

    # Run inference on the processed data
    preds = inference(model, X)

    # Convert the binary prediction to its corresponding label
    prediction = lb.inverse_transform(preds)[0]

    return {"prediction": prediction}
