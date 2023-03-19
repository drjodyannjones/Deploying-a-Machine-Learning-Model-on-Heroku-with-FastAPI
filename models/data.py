import pandas as pd
from sklearn.preprocessing import LabelEncoder, LabelBinarizer, StandardScaler

def process_data(df: pd.DataFrame, categorical_features: list, label: str, training: bool=True):
    """
    Preprocess data.

    Inputs
    ------
    df : pandas.DataFrame
        Data to be processed.
    categorical_features : list
        Names of the categorical features.
    label : str
        Name of the target variable.
    training : bool
        Whether the function is called for training or testing.
        When True, a new LabelEncoder and LabelBinarizer are fit on the data.
        When False, they are assumed to be fitted previously.

    Returns
    -------
    X : numpy.ndarray
        Processed data.
    y : numpy.ndarray
        Target variable.
    le : sklearn.preprocessing.LabelEncoder
        LabelEncoder object fitted on the target variable.
    lb : sklearn.preprocessing.LabelBinarizer
        LabelBinarizer object fitted on the target variable.
    scaler : sklearn.preprocessing.StandardScaler
        StandardScaler object fitted on the processed data.
    """

    X = df.drop(label, axis=1)
    y = df[label]

    if training:
        le = LabelEncoder()
        y = le.fit_transform(y)

        lb = LabelBinarizer()
        y = lb.fit_transform(y)

    else:
        le, lb = None, None

        y = le.transform(y)
        y = lb.transform(y)

    # Convert categorical variables to one-hot encoded variables
    for feature in categorical_features:
        X[feature] = X[feature].astype("category")

    X = pd.get_dummies(X, columns=categorical_features)

    # Scale numerical features
    scaler = StandardScaler()
    numerical_features = [
        col for col in X.columns if col not in categorical_features
    ]
    X[numerical_features] = scaler.fit_transform(X[numerical_features])

    return X.to_numpy(), y, le, lb, scaler

