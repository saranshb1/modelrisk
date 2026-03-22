"""Conditional VaR (Expected Shortfall) models."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


class CVaR:
    """Conditional Value at Risk (Expected Shortfall).

    CVaR is the expected loss given that the loss exceeds VaR. It is a coherent
    risk measure and preferred over VaR under Basel III/IV for internal model
    approaches.

    Parameters
    ----------
    confidence_level : float
        Confidence level, e.g. 0.975 for 97.5% ES (Basel III standard).
    holding_period : int
        Holding period in days.
    method : str
        One of 'historical', 'parametric', or 'montecarlo'.
    n_simulations : int
        Number of simulations (only used when method='montecarlo').
    random_state : int or None

    Examples
    --------
    >>> es = CVaR(confidence_level=0.975, method='historical')
    >>> es.fit(returns)
    >>> es.cvar()
    """

    METHODS = ("historical", "parametric", "montecarlo")

    def __init__(
        self,
        confidence_level: float = 0.975,
        holding_period: int = 1,
        method: str = "historical",
        n_simulations: int = 100_000,
        random_state: int | None = 42,
    ) -> None:
        if method not in self.METHODS:
            raise ValueError(f"method must be one of {self.METHODS}.")
        self.confidence_level = confidence_level
        self.holding_period = holding_period
        self.method = method
        self.n_simulations = n_simulations
        self.random_state = random_state
        self._returns: np.ndarray | None = None
        self.mu_: float | None = None
        self.sigma_: float | None = None

    def fit(self, returns: pd.Series | np.ndarray) -> "CVaR":
        """Fit the model on historical returns.

        Parameters
        ----------
        returns : array-like
            Daily P&L or log-returns (losses as negative values).
        """
        self._returns = np.asarray(returns, dtype=float)
        self.mu_ = float(np.mean(self._returns))
        self.sigma_ = float(np.std(self._returns, ddof=1))
        return self

    def cvar(self) -> float:
        """Compute CVaR / Expected Shortfall.

        Returns
        -------
        float : CVaR as a positive loss figure.
        """
        if self._returns is None:
            raise RuntimeError("Call fit() before cvar().")

        if self.method == "historical":
            return self._historical_cvar()
        elif self.method == "parametric":
            return self._parametric_cvar()
        else:
            return self._montecarlo_cvar()

    def _historical_cvar(self) -> float:
        threshold = np.quantile(self._returns, 1 - self.confidence_level)
        tail_losses = self._returns[self._returns <= threshold]
        if len(tail_losses) == 0:
            return float(-threshold * np.sqrt(self.holding_period))
        return float(-np.mean(tail_losses) * np.sqrt(self.holding_period))

    def _parametric_cvar(self) -> float:
        alpha = 1 - self.confidence_level
        z = stats.norm.ppf(alpha)
        phi_z = stats.norm.pdf(z)
        es = -(self.mu_ - self.sigma_ * phi_z / alpha)
        return float(es * np.sqrt(self.holding_period))

    def _montecarlo_cvar(self) -> float:
        rng = np.random.default_rng(self.random_state)
        sim = rng.normal(
            self.mu_ * self.holding_period,
            self.sigma_ * np.sqrt(self.holding_period),
            size=self.n_simulations,
        )
        threshold = np.quantile(sim, 1 - self.confidence_level)
        tail = sim[sim <= threshold]
        return float(-np.mean(tail))

    def var(self) -> float:
        """Return the VaR at the same confidence level (for comparison)."""
        if self._returns is None:
            raise RuntimeError("Call fit() before var().")
        q = np.quantile(self._returns, 1 - self.confidence_level)
        return float(-q * np.sqrt(self.holding_period))

    def summary(self) -> pd.Series:
        """Return a summary Series with VaR, CVaR, and model parameters."""
        return pd.Series(
            {
                "method": self.method,
                "confidence_level": self.confidence_level,
                "holding_period_days": self.holding_period,
                "var": self.var(),
                "cvar_es": self.cvar(),
                "cvar_var_ratio": self.cvar() / max(self.var(), 1e-9),
            }
        )
