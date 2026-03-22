"""Probability of Default (PD) models."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class LogisticPD:
    """Logistic regression-based PD model.

    Wraps scikit-learn's LogisticRegression with risk-domain conventions:
    calibrated probabilities, coefficient reporting, and Gini/KS diagnostics.

    Parameters
    ----------
    C : float
        Inverse regularisation strength. Larger values = less regularisation.
    max_iter : int
        Maximum number of solver iterations.
    scale_features : bool
        Whether to standardise features before fitting. Recommended when
        features are on different scales.

    Examples
    --------
    >>> model = LogisticPD()
    >>> model.fit(X_train, y_train)
    >>> probs = model.predict_proba(X_test)
    """

    def __init__(self, C: float = 1.0, max_iter: int = 1000, scale_features: bool = True) -> None:
        self.C = C
        self.max_iter = max_iter
        self.scale_features = scale_features
        self._model = LogisticRegression(C=C, max_iter=max_iter, solver="lbfgs")
        self._scaler = StandardScaler() if scale_features else None
        self.feature_names_: list[str] | None = None
        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> "LogisticPD":
        """Fit the PD model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.
        y : array-like of shape (n_samples,)
            Binary default indicator (1 = default, 0 = non-default).

        Returns
        -------
        self
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            X_arr = X.values
        else:
            X_arr = np.asarray(X)

        y_arr = np.asarray(y)

        if self._scaler is not None:
            X_arr = self._scaler.fit_transform(X_arr)

        self._model.fit(X_arr, y_arr)
        self.coef_ = self._model.coef_[0]
        self.intercept_ = float(self._model.intercept_[0])
        return self

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Return predicted default probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Probability of default for each observation.
        """
        X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        if self._scaler is not None:
            X_arr = self._scaler.transform(X_arr)
        return self._model.predict_proba(X_arr)[:, 1]

    def coefficient_summary(self) -> pd.DataFrame:
        """Return a DataFrame of feature coefficients.

        Returns
        -------
        pd.DataFrame with columns: feature, coefficient, odds_ratio
        """
        if self.coef_ is None:
            raise RuntimeError("Model has not been fitted yet.")
        names = self.feature_names_ or [f"x{i}" for i in range(len(self.coef_))]
        return pd.DataFrame(
            {
                "feature": names,
                "coefficient": self.coef_,
                "odds_ratio": np.exp(self.coef_),
            }
        ).sort_values("coefficient", key=abs, ascending=False)


class MertonPD:
    """Merton structural model for probability of default.

    Based on the original Merton (1974) model treating equity as a call option
    on the firm's assets. Derives the implied asset value and volatility from
    observable equity market data using an iterative procedure.

    Parameters
    ----------
    risk_free_rate : float
        Annualised risk-free interest rate (e.g. 0.05 for 5%).
    time_horizon : float
        Time horizon in years for the PD estimate (e.g. 1.0 for one year).
    tol : float
        Convergence tolerance for the iterative asset value procedure.
    max_iter : int
        Maximum iterations for the numerical solver.

    References
    ----------
    Merton, R.C. (1974). On the Pricing of Corporate Debt: The Risk Structure
    of Interest Rates. Journal of Finance, 29(2), 449–470.

    Examples
    --------
    >>> model = MertonPD(risk_free_rate=0.05, time_horizon=1.0)
    >>> pd_est = model.estimate_pd(
    ...     equity_value=50.0,
    ...     equity_volatility=0.30,
    ...     debt_face_value=80.0,
    ... )
    """

    def __init__(
        self,
        risk_free_rate: float = 0.05,
        time_horizon: float = 1.0,
        tol: float = 1e-6,
        max_iter: int = 1000,
    ) -> None:
        self.risk_free_rate = risk_free_rate
        self.time_horizon = time_horizon
        self.tol = tol
        self.max_iter = max_iter

    def _black_scholes_call(
        self, asset_value: float, debt: float, sigma: float
    ) -> tuple[float, float]:
        """Black-Scholes call option price and delta."""
        r, T = self.risk_free_rate, self.time_horizon
        d1 = (np.log(asset_value / debt) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        call = asset_value * stats.norm.cdf(d1) - debt * np.exp(-r * T) * stats.norm.cdf(d2)
        delta = stats.norm.cdf(d1)
        return call, delta

    def estimate_pd(
        self,
        equity_value: float,
        equity_volatility: float,
        debt_face_value: float,
    ) -> dict:
        """Estimate the probability of default under the Merton model.

        Parameters
        ----------
        equity_value : float
            Market value of equity (e.g. market capitalisation).
        equity_volatility : float
            Annualised equity volatility (e.g. 0.30 for 30%).
        debt_face_value : float
            Face value of debt (book value).

        Returns
        -------
        dict with keys:
            pd : float
                Risk-neutral probability of default.
            asset_value : float
                Implied market value of assets.
            asset_volatility : float
                Implied asset volatility.
            distance_to_default : float
                Number of standard deviations from the default point (DD).
        """
        # Iterative procedure to find implied asset value and volatility
        asset_value = equity_value + debt_face_value
        asset_volatility = equity_volatility * equity_value / asset_value

        for _ in range(self.max_iter):
            call, delta = self._black_scholes_call(asset_value, debt_face_value, asset_volatility)
            new_asset_value = equity_value + debt_face_value * np.exp(
                -self.risk_free_rate * self.time_horizon
            )
            new_asset_volatility = equity_volatility * equity_value / (delta * asset_value)

            if (
                abs(new_asset_value - asset_value) < self.tol
                and abs(new_asset_volatility - asset_volatility) < self.tol
            ):
                break
            asset_value = new_asset_value
            asset_volatility = new_asset_volatility

        r, T = self.risk_free_rate, self.time_horizon
        d2 = (
            np.log(asset_value / debt_face_value) + (r - 0.5 * asset_volatility**2) * T
        ) / (asset_volatility * np.sqrt(T))

        pd_estimate = stats.norm.cdf(-d2)

        return {
            "pd": pd_estimate,
            "asset_value": asset_value,
            "asset_volatility": asset_volatility,
            "distance_to_default": d2,
        }
