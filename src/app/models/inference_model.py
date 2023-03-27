from joblib import load
from .ml.data import process_data
from .ml.model import inference


def run_inference(data, cat_features):
    """
    Load model and run inference
    Parameters
    ----------
    root_path
    data
    cat_features

    Returns
    -------
    prediction
    """
    model = load("model/model.joblib")
    encoder = load("model/encoder.joblib")
    lb = load("model/lb.joblib")

    X, _, _, _ = process_data(
        data,
        categorical_features=cat_features,
        encoder=encoder, lb=lb, training=False)

    pred = inference(model, X)
    prediction = lb.inverse_transform(pred)[0]

    return prediction