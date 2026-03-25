"""Regression model evaluation metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn import metrics as skmetrics


class RegressionMetrics:
    """Regression metrics for LGD, EAD, and other continuous risk estimates.

    Covers standard metrics (RMSE, MAE, R²) plus risk-domain extensions
    such as mean absolute percentage error and error distribution analysis.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Observed values.
    y_pred : array-like of shape (n_samples,)
        Predicted values.

    Examples
    --------
    >>> rm = RegressionMetrics(y_true, y_pred)
    >>> rm.summary()
    """

    def __init__(
        self,
        y_true: pd.Series | np.ndarray,
        y_pred: pd.Series | np.ndarray,
    ) -> None:
        self.y_true = np.asarray(y_true, dtype=float)
        self.y_pred = np.asarray(y_pred, dtype=float)
        if len(self.y_true) != len(self.y_pred):
            raise ValueError("y_true and y_pred must have the same length.")

    def rmse(self) -> float:
        """Root Mean Squared Error."""
        return float(np.sqrt(skmetrics.mean_squared_error(self.y_true, self.y_pred)))

    def mse(self) -> float:
        """Mean Squared Error."""
        return float(skmetrics.mean_squared_error(self.y_true, self.y_pred))

    def mae(self) -> float:
        """Mean Absolute Error."""
        return float(skmetrics.mean_absolute_error(self.y_true, self.y_pred))

    def r_squared(self) -> float:
        """Coefficient of determination R²."""
        return float(skmetrics.r2_score(self.y_true, self.y_pred))

    def adjusted_r_squared(self, n_features: int) -> float:
        """Adjusted R² accounting for number of predictors.

        Parameters
        ----------
        n_features : int
            Number of features in the model.
        """
        n = len(self.y_true)
        r2 = self.r_squared()
        return float(1 - (1 - r2) * (n - 1) / (n - n_features - 1))

    def mape(self) -> float:
        """Mean Absolute Percentage Error.

        Note: undefined where y_true = 0. Those observations are skipped.
        """
        mask = self.y_true != 0
        if not mask.any():
            return float("nan")
        return float(
            np.mean(np.abs((self.y_true[mask] - self.y_pred[mask]) / self.y_true[mask])) * 100
        )

    def median_absolute_error(self) -> float:
        """Median Absolute Error — robust to outliers."""
        return float(skmetrics.median_absolute_error(self.y_true, self.y_pred))

    def max_error(self) -> float:
        """Maximum absolute prediction error."""
        return float(skmetrics.max_error(self.y_true, self.y_pred))

    def mean_bias(self) -> float:
        """Mean prediction bias (positive = over-prediction)."""
        return float(np.mean(self.y_pred - self.y_true))

    def error_percentiles(self) -> pd.Series:
        """Return percentiles of the absolute error distribution."""
        abs_errors = np.abs(self.y_pred - self.y_true)
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        return pd.Series(
            np.percentile(abs_errors, percentiles),
            index=[f"p{p}" for p in percentiles],
            name="absolute_error_percentiles",
        )

    def summary(self, n_features: int = 1) -> pd.DataFrame:
        """Return all regression metrics as a tidy DataFrame."""
        rows = [
            ("RMSE", self.rmse(), "Root mean squared error; penalises large errors"),
            ("MSE", self.mse(), "Mean squared error"),
            ("MAE", self.mae(), "Mean absolute error; robust to outliers"),
            ("R²", self.r_squared(), "Explained variance; 1.0 = perfect"),
            (
                "Adjusted R²",
                self.adjusted_r_squared(n_features),
                "R² penalised for model complexity",
            ),
            ("MAPE (%)", self.mape(), "Mean absolute percentage error"),
            ("Median AE", self.median_absolute_error(), "Median absolute error"),
            ("Max error", self.max_error(), "Worst-case absolute prediction error"),
            ("Mean bias", self.mean_bias(), "Systematic over/under-prediction"),
        ]
        return pd.DataFrame(rows, columns=["metric", "value", "interpretation"])
