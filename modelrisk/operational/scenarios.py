"""Scenario analysis and Extreme Value Theory for operational risk."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


class ScenarioAnalysis:
    """Scenario-based operational risk assessment.

    Combines expert-elicited scenario estimates with internal loss data
    using a weighted aggregation approach. Each scenario is defined by
    a frequency estimate and a severity distribution.

    Examples
    --------
    >>> sa = ScenarioAnalysis()
    >>> sa.add_scenario("cyber breach", frequency=0.2, severity_mean=500_000, severity_std=200_000)
    >>> sa.add_scenario("rogue trader", frequency=0.05,
            severity_mean=5_000_000, severity_std=2_000_000)
    >>> sa.expected_annual_loss()
    """

    def __init__(self, n_simulations: int = 100_000, random_state: int | None = 42) -> None:
        self.n_simulations = n_simulations
        self.random_state = random_state
        self._scenarios: list[dict] = []

    def add_scenario(
        self,
        name: str,
        frequency: float,
        severity_mean: float,
        severity_std: float,
        severity_dist: str = "lognormal",
    ) -> ScenarioAnalysis:
        """Add a risk scenario.

        Parameters
        ----------
        name : str
            Scenario identifier.
        frequency : float
            Expected annual frequency (Poisson lambda).
        severity_mean : float
            Mean loss per event.
        severity_std : float
            Standard deviation of loss per event.
        severity_dist : str
            Distribution: 'lognormal' or 'gamma'.

        Returns
        -------
        self
        """
        if severity_dist == "lognormal":
            sigma2 = np.log(1 + (severity_std / severity_mean) ** 2)
            mu = np.log(severity_mean) - 0.5 * sigma2
            params = {"mu": mu, "sigma": np.sqrt(sigma2)}
        elif severity_dist == "gamma":
            shape = (severity_mean / severity_std) ** 2
            scale = severity_std**2 / severity_mean
            params = {"shape": shape, "scale": scale}
        else:
            raise ValueError("severity_dist must be 'lognormal' or 'gamma'.")

        self._scenarios.append(
            {
                "name": name,
                "frequency": frequency,
                "severity_mean": severity_mean,
                "severity_std": severity_std,
                "severity_dist": severity_dist,
                "params": params,
            }
        )
        return self

    def expected_annual_loss(self) -> pd.DataFrame:
        """Compute expected annual loss per scenario.

        Returns
        -------
        pd.DataFrame with scenario-level EAL and totals.
        """
        rows = [
            {
                "scenario": s["name"],
                "frequency": s["frequency"],
                "severity_mean": s["severity_mean"],
                "expected_annual_loss": s["frequency"] * s["severity_mean"],
            }
            for s in self._scenarios
        ]
        df = pd.DataFrame(rows)
        total = pd.DataFrame(
            [{"scenario": "TOTAL", "frequency": df["frequency"].sum(),
                "severity_mean": np.nan,
                "expected_annual_loss": df["expected_annual_loss"].sum()}]
        )
        return pd.concat([df, total], ignore_index=True)

    def simulate(self) -> np.ndarray:
        """Monte Carlo simulation across all scenarios.

        Returns
        -------
        np.ndarray of shape (n_simulations,) — aggregate annual losses.
        """
        rng = np.random.default_rng(self.random_state)
        aggregate = np.zeros(self.n_simulations)
        for s in self._scenarios:
            counts = rng.poisson(s["frequency"], size=self.n_simulations)
            total_events = int(counts.sum())
            if total_events == 0:
                continue
            if s["severity_dist"] == "lognormal":
                draws = rng.lognormal(s["params"]["mu"], s["params"]["sigma"], size=total_events)
            else:
                draws = rng.gamma(s["params"]["shape"], s["params"]["scale"], size=total_events)
            idx = np.repeat(np.arange(self.n_simulations), counts)
            np.add.at(aggregate, idx, draws)
        return aggregate


class ExtremeValueModel:
    """Generalised Pareto Distribution (GPD) for tail risk — Peaks Over Threshold.

    Fits a GPD to losses exceeding a chosen threshold, consistent with
    Extreme Value Theory (EVT). Used to extrapolate into the far tail
    beyond available data.

    Parameters
    ----------
    threshold : float or None
        Loss threshold u. If None, the 90th percentile of the data is used.

    References
    ----------
    McNeil, A.J., Frey, R., Embrechts, P. (2015). Quantitative Risk Management.
    Princeton University Press. Chapter 5.

    Examples
    --------
    >>> evt = ExtremeValueModel(threshold=100_000)
    >>> evt.fit(losses)
    >>> evt.var(0.999)
    >>> evt.cvar(0.999)
    """

    def __init__(self, threshold: float | None = None) -> None:
        self.threshold = threshold
        self.xi_: float | None = None   # shape
        self.beta_: float | None = None  # scale
        self._n_total: int | None = None
        self._n_exceedances: int | None = None
        self._threshold_used: float | None = None

    def fit(self, losses: pd.Series | np.ndarray) -> ExtremeValueModel:
        """Fit the GPD to exceedances above the threshold.

        Parameters
        ----------
        losses : array-like of positive loss values.

        Returns
        -------
        self
        """
        x = np.asarray(losses, dtype=float)
        self._n_total = len(x)

        if self.threshold is None:
            self._threshold_used = float(np.quantile(x, 0.90))
        else:
            self._threshold_used = float(self.threshold)

        exceedances = x[x > self._threshold_used] - self._threshold_used
        self._n_exceedances = len(exceedances)

        if self._n_exceedances < 10:
            raise ValueError(
                f"Only {self._n_exceedances} exceedances above threshold "
                f"{self._threshold_used:.2f}. Reduce the threshold."
            )

        xi, _, beta = stats.genpareto.fit(exceedances, floc=0)
        self.xi_ = float(xi)
        self.beta_ = float(beta)
        return self

    def var(self, confidence_level: float = 0.999) -> float:
        """GPD-based VaR estimate via EVT.

        Parameters
        ----------
        confidence_level : float

        Returns
        -------
        float : VaR at the specified confidence level.
        """
        if self.xi_ is None:
            raise RuntimeError("Call fit() first.")
        u = self._threshold_used
        n, nu = self._n_total, self._n_exceedances
        p = confidence_level
        if self.xi_ == 0:
            return float(u + self.beta_ * np.log(n / nu * (1 - p)))
        return float(u + self.beta_ / self.xi_ * ((n / nu * (1 - p)) ** (-self.xi_) - 1))

    def cvar(self, confidence_level: float = 0.999) -> float:
        """GPD-based CVaR (Expected Shortfall) via EVT.

        Parameters
        ----------
        confidence_level : float

        Returns
        -------
        float
        """
        if self.xi_ is None:
            raise RuntimeError("Call fit() first.")
        v = self.var(confidence_level)
        u = self._threshold_used
        if self.xi_ >= 1:
            return float("inf")
        return float((v + self.beta_ - self.xi_ * u) / (1 - self.xi_))

    def tail_summary(self) -> pd.Series:
        """Return fitted GPD parameters and diagnostics."""
        if self.xi_ is None:
            raise RuntimeError("Call fit() first.")
        return pd.Series(
            {
                "threshold": self._threshold_used,
                "n_total": self._n_total,
                "n_exceedances": self._n_exceedances,
                "exceedance_rate": self._n_exceedances / self._n_total,
                "xi_shape": self.xi_,
                "beta_scale": self.beta_,
                "var_99_9": self.var(0.999),
                "cvar_99_9": self.cvar(0.999),
            }
        )
