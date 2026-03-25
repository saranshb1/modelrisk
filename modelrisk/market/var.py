"""Value at Risk (VaR) models."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


class HistoricalVaR:
    """Historical simulation VaR.

    Estimates VaR from the empirical distribution of historical P&L or returns,
    making no distributional assumptions.

    Parameters
    ----------
    confidence_level : float
        Confidence level, e.g. 0.99 for 99% VaR.
    holding_period : int
        Holding period in days. VaR is scaled by sqrt(holding_period).

    Examples
    --------
    >>> var_model = HistoricalVaR(confidence_level=0.99)
    >>> var_model.fit(returns)
    >>> var_model.var()
    """

    def __init__(self, confidence_level: float = 0.99, holding_period: int = 1) -> None:
        if not 0 < confidence_level < 1:
            raise ValueError("confidence_level must be in (0, 1).")
        self.confidence_level = confidence_level
        self.holding_period = holding_period
        self._returns: np.ndarray | None = None

    def fit(self, returns: pd.Series | np.ndarray) -> HistoricalVaR:
        """Store the historical return series.

        Parameters
        ----------
        returns : array-like
            Daily P&L or log-returns (losses as negative values).

        Returns
        -------
        self
        """
        self._returns = np.asarray(returns, dtype=float)
        return self

    def var(self) -> float:
        """Compute VaR as a positive loss figure.

        Returns
        -------
        float : VaR at the specified confidence level and holding period.
        """
        if self._returns is None:
            raise RuntimeError("Call fit() before var().")
        quantile = np.quantile(self._returns, 1 - self.confidence_level)
        return float(-quantile * np.sqrt(self.holding_period))

    def var_series(self, window: int = 250) -> pd.Series:
        """Compute rolling VaR over time.

        Parameters
        ----------
        window : int
            Rolling window in days.

        Returns
        -------
        pd.Series of rolling VaR estimates.
        """
        if self._returns is None:
            raise RuntimeError("Call fit() before var_series().")
        results = []
        for i in range(window, len(self._returns) + 1):
            sub = self._returns[i - window: i]
            q = np.quantile(sub, 1 - self.confidence_level)
            results.append(-q * np.sqrt(self.holding_period))
        return pd.Series(results, name=f"VaR_{int(self.confidence_level * 100)}")


class ParametricVaR:
    """Parametric (variance-covariance) VaR under a Normal distribution.

    Parameters
    ----------
    confidence_level : float
    holding_period : int

    Examples
    --------
    >>> var_model = ParametricVaR(confidence_level=0.99)
    >>> var_model.fit(returns)
    >>> var_model.var()
    """

    def __init__(self, confidence_level: float = 0.99, holding_period: int = 1) -> None:
        self.confidence_level = confidence_level
        self.holding_period = holding_period
        self.mu_: float | None = None
        self.sigma_: float | None = None

    def fit(self, returns: pd.Series | np.ndarray) -> ParametricVaR:
        r = np.asarray(returns, dtype=float)
        self.mu_ = float(np.mean(r))
        self.sigma_ = float(np.std(r, ddof=1))
        return self

    def var(self) -> float:
        if self.sigma_ is None:
            raise RuntimeError("Call fit() before var().")
        z = stats.norm.ppf(1 - self.confidence_level)
        daily_var = -(self.mu_ + z * self.sigma_)
        return float(daily_var * np.sqrt(self.holding_period))

    def var_with_t(self, df: float = 5.0) -> float:
        """VaR under a Student-t distribution (heavier tails).

        Parameters
        ----------
        df : float
            Degrees of freedom for the t-distribution.

        Returns
        -------
        float
        """
        if self.sigma_ is None:
            raise RuntimeError("Call fit() before var_with_t().")
        t_quantile = stats.t.ppf(1 - self.confidence_level, df=df)
        daily_var = -(self.mu_ + t_quantile * self.sigma_)
        return float(daily_var * np.sqrt(self.holding_period))


class MonteCarloVaR:
    """Monte Carlo simulation VaR.

    Simulates future P&L paths from estimated return parameters and computes
    VaR from the resulting loss distribution.

    Parameters
    ----------
    confidence_level : float
    holding_period : int
    n_simulations : int
        Number of Monte Carlo paths.
    random_state : int or None

    Examples
    --------
    >>> var_model = MonteCarloVaR(n_simulations=100_000)
    >>> var_model.fit(returns)
    >>> var_model.var()
    """

    def __init__(
        self,
        confidence_level: float = 0.99,
        holding_period: int = 1,
        n_simulations: int = 100_000,
        random_state: int | None = 42,
    ) -> None:
        self.confidence_level = confidence_level
        self.holding_period = holding_period
        self.n_simulations = n_simulations
        self.random_state = random_state
        self.mu_: float | None = None
        self.sigma_: float | None = None

    def fit(self, returns: pd.Series | np.ndarray) -> MonteCarloVaR:
        r = np.asarray(returns, dtype=float)
        self.mu_ = float(np.mean(r))
        self.sigma_ = float(np.std(r, ddof=1))
        return self

    def var(self) -> float:
        if self.sigma_ is None:
            raise RuntimeError("Call fit() before var().")
        rng = np.random.default_rng(self.random_state)
        simulated = rng.normal(
            self.mu_ * self.holding_period,
            self.sigma_ * np.sqrt(self.holding_period),
            size=self.n_simulations,
        )
        return float(-np.quantile(simulated, 1 - self.confidence_level))

    @property
    def simulated_losses_(self) -> np.ndarray:
        """Return the simulated loss distribution (positive = loss)."""
        if self.sigma_ is None:
            raise RuntimeError("Call fit() first.")
        rng = np.random.default_rng(self.random_state)
        return -rng.normal(
            self.mu_ * self.holding_period,
            self.sigma_ * np.sqrt(self.holding_period),
            size=self.n_simulations,
        )
