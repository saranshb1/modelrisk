"""Shared pytest fixtures for modelrisk test suite."""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification


@pytest.fixture(scope="session")
def binary_classification_data():
    """Binary classification dataset: (X_train, X_test, y_train, y_test)."""
    X, y = make_classification(
        n_samples=1000, n_features=10, n_informative=5,
        n_redundant=2, random_state=42
    )
    split = 800
    return (
        pd.DataFrame(X[:split], columns=[f"f{i}" for i in range(10)]),
        pd.DataFrame(X[split:], columns=[f"f{i}" for i in range(10)]),
        pd.Series(y[:split]),
        pd.Series(y[split:]),
    )


@pytest.fixture(scope="session")
def daily_returns():
    """250 days of simulated daily returns."""
    rng = np.random.default_rng(0)
    return pd.Series(rng.normal(0.0002, 0.015, size=250), name="returns")


@pytest.fixture(scope="session")
def loss_data():
    """Simulated operational loss amounts."""
    rng = np.random.default_rng(1)
    return pd.Series(rng.lognormal(mean=10, sigma=1.5, size=500))


@pytest.fixture(scope="session")
def fitted_logistic_pd(binary_classification_data):
    """Pre-fitted LogisticPD model."""
    from modelrisk.credit.pd import LogisticPD
    X_train, _, y_train, _ = binary_classification_data
    model = LogisticPD()
    model.fit(X_train, y_train)
    return model


@pytest.fixture(scope="session")
def predicted_probabilities(fitted_logistic_pd, binary_classification_data):
    """Predicted probabilities and ground truth for evaluation tests."""
    _, X_test, _, y_test = binary_classification_data
    proba = fitted_logistic_pd.predict_proba(X_test)
    return y_test.values, proba
