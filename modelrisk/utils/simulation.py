"""Monte Carlo simulation engine."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


class MonteCarloEngine:
    """General-purpose Monte Carlo simulation engine.

    Supports correlated multi-asset simulations via Cholesky decomposition,
    geometric Brownian motion paths, and customisable payoff functions.

    Parameters
    ----------
    n_simulations : int
    random_state : int or None

    Examples
    --------
    >>> engine = MonteCarloEngine(n_simulations=100_000)
    >>> losses = engine.simulate_losses(mean=0.0, std=0.02, horizon=10)
    >>> paths = engine.gbm_paths(S0=100, mu=0.05, sigma=0.20, T=1.0, steps=252)
    """

    def __init__(self, n_simulations: int = 100_000, random_state: int | None = 42) -> None:
        self.n_simulations = n_simulations
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)

    def simulate_losses(
        self,
        mean: float,
        std: float,
        horizon: int = 1,
        distribution: str = "normal",
    ) -> np.ndarray:
        """Simulate loss distribution over a time horizon.

        Parameters
        ----------
        mean : float — daily mean return.
        std : float — daily standard deviation.
        horizon : int — holding period in days.
        distribution : str — 'normal' or 't5' (Student-t with 5 df).

        Returns
        -------
        np.ndarray of shape (n_simulations,) — simulated losses (positive = loss).
        """
        if distribution == "normal":
            draws = self._rng.normal(
                mean * horizon, std * np.sqrt(horizon), size=self.n_simulations
            )
        elif distribution == "t5":
            from scipy import stats
            draws = (
                stats.t.rvs(5, size=self.n_simulations, 
                                random_state=int(self._rng.integers(0, 2**31)))
                * std * np.sqrt(horizon)
                + mean * horizon
            )
        else:
            raise ValueError(f"Unknown distribution '{distribution}'. Use 'normal' or 't5'.")
        return -draws  # flip sign: positive = loss

    def gbm_paths(
        self,
        S0: float,
        mu: float,
        sigma: float,
        T: float = 1.0,
        steps: int = 252,
    ) -> np.ndarray:
        """Simulate Geometric Brownian Motion asset price paths.

        Parameters
        ----------
        S0 : float — initial price.
        mu : float — annualised drift.
        sigma : float — annualised volatility.
        T : float — time horizon in years.
        steps : int — number of time steps.

        Returns
        -------
        np.ndarray of shape (n_simulations, steps + 1).
        """
        dt = T / steps
        paths = np.zeros((self.n_simulations, steps + 1))
        paths[:, 0] = S0
        noise = self._rng.standard_normal((self.n_simulations, steps))
        increments = np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * noise)
        paths[:, 1:] = S0 * np.cumprod(increments, axis=1)
        return paths

    def correlated_normals(
        self, correlation_matrix: np.ndarray, n_assets: int | None = None
    ) -> np.ndarray:
        """Simulate correlated standard normal draws via Cholesky decomposition.

        Parameters
        ----------
        correlation_matrix : np.ndarray of shape (n_assets, n_assets)
        n_assets : int or None — inferred from matrix if None.

        Returns
        -------
        np.ndarray of shape (n_simulations, n_assets).
        """
        corr = np.asarray(correlation_matrix)
        chol = np.linalg.cholesky(corr)
        draws = self._rng.standard_normal((self.n_simulations, corr.shape[0]))
        return draws @ chol.T

    def percentile_summary(self, simulated: np.ndarray) -> pd.Series:
        """Return percentile summary of a simulated distribution."""
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99, 99.5, 99.9]
        return pd.Series(
            np.percentile(simulated, percentiles),
            index=[f"p{p}" for p in percentiles],
            name="simulated_distribution",
        )
