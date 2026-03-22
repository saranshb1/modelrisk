"""Volatility models: EWMA and GARCH(1,1)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize


class EWMAVolatility:
    """Exponentially Weighted Moving Average (EWMA) volatility model.

    Applies exponentially declining weights to squared returns, giving more
    weight to recent observations. This is the RiskMetrics approach.

    Parameters
    ----------
    lambda_ : float
        Decay factor (0 < lambda_ < 1). RiskMetrics standard: 0.94 for daily.

    Examples
    --------
    >>> model = EWMAVolatility(lambda_=0.94)
    >>> model.fit(returns)
    >>> model.volatility_series()
    """

    def __init__(self, lambda_: float = 0.94) -> None:
        if not 0 < lambda_ < 1:
            raise ValueError("lambda_ must be in (0, 1).")
        self.lambda_ = lambda_
        self._returns: np.ndarray | None = None
        self._variance_series: np.ndarray | None = None

    def fit(self, returns: pd.Series | np.ndarray) -> "EWMAVolatility":
        """Fit the EWMA model.

        Parameters
        ----------
        returns : array-like of daily returns.

        Returns
        -------
        self
        """
        r = np.asarray(returns, dtype=float)
        self._returns = r
        n = len(r)
        variances = np.zeros(n)
        variances[0] = r[0] ** 2
        for t in range(1, n):
            variances[t] = self.lambda_ * variances[t - 1] + (1 - self.lambda_) * r[t - 1] ** 2
        self._variance_series = variances
        return self

    def volatility_series(self, annualise: bool = True) -> pd.Series:
        """Return the full volatility series.

        Parameters
        ----------
        annualise : bool
            If True, multiply by sqrt(252) to annualise.

        Returns
        -------
        pd.Series of daily or annualised volatilities.
        """
        if self._variance_series is None:
            raise RuntimeError("Call fit() first.")
        vol = np.sqrt(self._variance_series)
        if annualise:
            vol = vol * np.sqrt(252)
        return pd.Series(vol, name="ewma_volatility")

    def current_volatility(self, annualise: bool = True) -> float:
        """Return the most recent volatility estimate."""
        if self._variance_series is None:
            raise RuntimeError("Call fit() first.")
        vol = float(np.sqrt(self._variance_series[-1]))
        return vol * np.sqrt(252) if annualise else vol

    def forecast(self, horizon: int = 1) -> float:
        """Forecast volatility h steps ahead.

        Under EWMA, the multi-step forecast equals the current estimate
        (variance is a random walk in this model).

        Parameters
        ----------
        horizon : int
            Forecast horizon in days.

        Returns
        -------
        float : annualised volatility forecast.
        """
        if self._variance_series is None:
            raise RuntimeError("Call fit() first.")
        current_var = self._variance_series[-1]
        return float(np.sqrt(current_var * horizon / horizon) * np.sqrt(252))


class GARCHVolatility:
    """GARCH(1,1) volatility model estimated by maximum likelihood.

    The canonical GARCH(1,1) specification:
        sigma^2_t = omega + alpha * epsilon^2_{t-1} + beta * sigma^2_{t-1}

    Estimated by maximising the Gaussian log-likelihood. Forecasts mean-revert
    to the unconditional variance omega / (1 - alpha - beta).

    Parameters
    ----------
    starting_params : tuple or None
        Initial (omega, alpha, beta) for the optimiser. If None, sensible
        defaults are used.

    Examples
    --------
    >>> model = GARCHVolatility()
    >>> model.fit(returns)
    >>> model.forecast(horizon=10)
    """

    def __init__(self, starting_params: tuple[float, float, float] | None = None) -> None:
        self.starting_params = starting_params or (1e-5, 0.1, 0.85)
        self.omega_: float | None = None
        self.alpha_: float | None = None
        self.beta_: float | None = None
        self._variance_series: np.ndarray | None = None
        self._returns: np.ndarray | None = None

    def _neg_log_likelihood(self, params: np.ndarray, returns: np.ndarray) -> float:
        omega, alpha, beta = params
        n = len(returns)
        sigma2 = np.zeros(n)
        sigma2[0] = np.var(returns)
        for t in range(1, n):
            sigma2[t] = omega + alpha * returns[t - 1] ** 2 + beta * sigma2[t - 1]
            sigma2[t] = max(sigma2[t], 1e-10)
        ll = -0.5 * np.sum(np.log(2 * np.pi * sigma2) + returns**2 / sigma2)
        return -ll

    def fit(self, returns: pd.Series | np.ndarray) -> "GARCHVolatility":
        """Estimate GARCH(1,1) parameters by MLE.

        Parameters
        ----------
        returns : array-like of daily returns.

        Returns
        -------
        self
        """
        r = np.asarray(returns, dtype=float)
        self._returns = r

        bounds = [(1e-9, None), (1e-6, 0.999), (1e-6, 0.999)]
        constraints = [{"type": "ineq", "fun": lambda p: 0.999 - p[1] - p[2]}]

        result = optimize.minimize(
            self._neg_log_likelihood,
            x0=np.array(self.starting_params),
            args=(r,),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 2000},
        )
        self.omega_, self.alpha_, self.beta_ = result.x

        n = len(r)
        sigma2 = np.zeros(n)
        sigma2[0] = np.var(r)
        for t in range(1, n):
            sigma2[t] = self.omega_ + self.alpha_ * r[t - 1] ** 2 + self.beta_ * sigma2[t - 1]
            sigma2[t] = max(sigma2[t], 1e-10)
        self._variance_series = sigma2
        return self

    def volatility_series(self, annualise: bool = True) -> pd.Series:
        """Return the conditional volatility series."""
        if self._variance_series is None:
            raise RuntimeError("Call fit() first.")
        vol = np.sqrt(self._variance_series)
        if annualise:
            vol = vol * np.sqrt(252)
        return pd.Series(vol, name="garch_volatility")

    def forecast(self, horizon: int = 10, annualise: bool = True) -> np.ndarray:
        """Multi-step ahead variance forecast.

        GARCH(1,1) forecasts mean-revert to the unconditional variance
        sigma_inf^2 = omega / (1 - alpha - beta).

        Parameters
        ----------
        horizon : int
            Number of days ahead.
        annualise : bool

        Returns
        -------
        np.ndarray of shape (horizon,) — daily variance forecasts.
        """
        if self.omega_ is None:
            raise RuntimeError("Call fit() first.")
        unconditional = self.omega_ / (1 - self.alpha_ - self.beta_)
        current_var = self._variance_series[-1]
        persistence = self.alpha_ + self.beta_

        forecasts = np.zeros(horizon)
        for h in range(1, horizon + 1):
            forecasts[h - 1] = unconditional + persistence ** (h - 1) * (
                current_var - unconditional
            )

        vol_forecasts = np.sqrt(forecasts)
        if annualise:
            vol_forecasts = vol_forecasts * np.sqrt(252)
        return vol_forecasts

    @property
    def unconditional_volatility(self) -> float:
        """Long-run unconditional volatility (annualised)."""
        if self.omega_ is None:
            raise RuntimeError("Call fit() first.")
        return float(np.sqrt(self.omega_ / (1 - self.alpha_ - self.beta_)) * np.sqrt(252))

    def parameter_summary(self) -> pd.Series:
        """Return estimated parameters."""
        if self.omega_ is None:
            raise RuntimeError("Call fit() first.")
        return pd.Series(
            {
                "omega": self.omega_,
                "alpha": self.alpha_,
                "beta": self.beta_,
                "persistence (alpha+beta)": self.alpha_ + self.beta_,
                "unconditional_vol_annualised": self.unconditional_volatility,
            }
        )
