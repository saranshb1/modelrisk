"""Loss Given Default (LGD) models."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize, stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


class BetaLGD:
    """Beta regression model for LGD estimation.

    LGD values are bounded in [0, 1], making the Beta distribution a natural
    choice. Fits a Beta regression by maximum likelihood estimation, with a
    logistic link function mapping linear predictors to the (0, 1) interval.

    Parameters
    ----------
    fit_intercept : bool
        Whether to include an intercept term.

    Examples
    --------
    >>> model = BetaLGD()
    >>> model.fit(X_train, lgd_train)
    >>> predictions = model.predict(X_test)
    """

    def __init__(self, fit_intercept: bool = True) -> None:
        self.fit_intercept = fit_intercept
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self.phi_: float | None = None  # precision parameter
        self.feature_names_: list[str] | None = None

    def _logistic(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    def _neg_log_likelihood(self, params: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        n_features = X.shape[1]
        beta = params[:n_features]
        log_phi = params[n_features]
        phi = np.exp(log_phi)

        mu = self._logistic(X @ beta)
        mu = np.clip(mu, 1e-6, 1 - 1e-6)
        y = np.clip(y, 1e-6, 1 - 1e-6)

        a = mu * phi
        b = (1 - mu) * phi

        ll = (
            stats.beta.logpdf(y, a, b).sum()
        )
        return -ll

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> BetaLGD:
        """Fit the Beta LGD model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
            LGD values in [0, 1].

        Returns
        -------
        self
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            X_arr = X.values.astype(float)
        else:
            X_arr = np.asarray(X, dtype=float)

        y_arr = np.asarray(y, dtype=float)

        if self.fit_intercept:
            X_arr = np.column_stack([np.ones(len(X_arr)), X_arr])

        n_params = X_arr.shape[1]
        x0 = np.zeros(n_params + 1)
        x0[-1] = np.log(5.0)  # initial log-phi

        result = optimize.minimize(
            self._neg_log_likelihood,
            x0,
            args=(X_arr, y_arr),
            method="L-BFGS-B",
            options={"maxiter": 1000},
        )

        if self.fit_intercept:
            self.intercept_ = result.x[0]
            self.coef_ = result.x[1:n_params]
        else:
            self.coef_ = result.x[:n_params]

        self.phi_ = float(np.exp(result.x[-1]))
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict mean LGD.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        np.ndarray of shape (n_samples,)
        """
        if self.coef_ is None:
            raise RuntimeError("Model has not been fitted yet.")
        X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=float)
        linear = X_arr @ self.coef_ + self.intercept_
        return self._logistic(linear)


class LinearLGD:
    """Ordinary least squares LGD model with clipping to [0, 1].

    A simple baseline model. Predictions are clipped to [0, 1] to remain
    valid as LGD estimates.

    Parameters
    ----------
    fit_intercept : bool
    scale_features : bool

    Examples
    --------
    >>> model = LinearLGD()
    >>> model.fit(X_train, lgd_train)
    >>> predictions = model.predict(X_test)
    """

    def __init__(self, fit_intercept: bool = True, scale_features: bool = False) -> None:
        self.fit_intercept = fit_intercept
        self.scale_features = scale_features
        self._model = LinearRegression(fit_intercept=fit_intercept)
        self._scaler = StandardScaler() if scale_features else None
        self.feature_names_: list[str] | None = None

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> LinearLGD:
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            X_arr = X.values
        else:
            X_arr = np.asarray(X)

        if self._scaler:
            X_arr = self._scaler.fit_transform(X_arr)

        self._model.fit(X_arr, np.asarray(y))
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        if self._scaler:
            X_arr = self._scaler.transform(X_arr)
        return np.clip(self._model.predict(X_arr), 0.0, 1.0)
