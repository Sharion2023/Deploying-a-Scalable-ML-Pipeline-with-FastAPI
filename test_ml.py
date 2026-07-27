import pytest
import numpy as np
from train_model import (
    train_model,
    compute_model_metrics,
)
from ml.model import inference
from sklearn.ensemble import RandomForestClassifier

# TODO: implement the first test. Change the function name and input as needed
def test_train_model_returns_correct_type():
    """
    Creating a random data set and training it to ensure train model returns a RandomForest Classifier
    """
    X = np.random.rand(20, 5)
    y = np.random.randint(0,2,20)
    model = train_model(X, y)

    assert isinstance (model, RandomForestClassifier)


# TODO: implement the second test. Change the function name and input as needed
def test_compute_model_metrics():
    """
    Testing to see whether the compute model matrix function returns values within range
    """
    y_pred = np.array([0,1,0,1,1])
    y_true = np.array([0,0,1,0,0])

    p, r, fb = compute_model_metrics(y_true, y_pred)

    assert 0 <= p <= 1
    assert 0 <= r <= 1
    assert 0 <= fb <= 1


# TODO: implement the third test. Change the function name and input as needed
def test_inference_type_and_shape():
    """
    Testing that the inference produces a valid array and that it outputs the correct
    of predictions
    """
    X_train = np.random.rand(20, 5)
    y_train = np.random.randint(0,2,20)

    model = train_model(X_train, y_train)
    preds = inference(model, X_train)

    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == X_train.shape[0]

