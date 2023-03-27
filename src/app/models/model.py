import logging
from numpy import mean
from numpy import std

from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

from .data import process_data


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    scores = cross_val_score(model, X_train, y_train, scoring='accuracy',
                             cv=cv, n_jobs=-1)
    logging.info('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision,
    recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, x):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(x)
    return preds


def compute_score_per_slice(trained_model, test, encoder,
                            lb, cat_features, root_path):
    """
    Compute score per category class slice
    Parameters
    ----------
    trained_model
    test
    encoder
    lb

    Returns
    -------

    """
    with open(f'{root_path}/model/slice_output.txt', 'w') as file:
        for category in cat_features:
            for cls in test[category].unique():
                temp_df = test[test[category] == cls]

                x_test, y_test, _, _ = process_data(
                    temp_df,
                    categorical_features=cat_features, training=False,
                    label="salary", encoder=encoder, lb=lb)

                y_pred = trained_model.predict(x_test)

                prc, rcl, fb = compute_model_metrics(y_test, y_pred)

                metric_info = "[%s]-[%s] Precision: %s " \
                              "Recall: %s FBeta: %s" % (category, cls,
                                                        prc, rcl, fb)
                logging.info(metric_info)
                file.write(metric_info + '\n')