"""Loss Distribution Approach (LDA) for operational risk capital."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


class LossDistributionApproach:
    """LDA operational risk capital model.

    Separately fits frequency and severity distributions, then combines them
    via Monte Carlo convolution to estimate the aggregate annual loss
    distribution and regulatory capital (99.9% VaR / CVaR).

    Parameters
    ----------
    frequency_dist : str
        Distribution for loss frequency. One of 'poisson', 'negative_binomial'.
    severity_dist : str
        Distribution for loss severity. One of 'lognormal', 'gamma', 'pareto'.
    n_simulations : int
        Monte Carlo paths for convolution.
    confidence_level : float
        For capital calculation (regulatory standard: 0.999).
    random_state : int or None

    Examples
    --------
    >>> lda = LossDistributionApproach()
    >>> lda.fit(frequencies, severities)
    >>> lda.capital_estimate()
    """

    FREQ_DISTS = ("poisson", "negative_binomial")
    SEV_DISTS = ("lognormal", "gamma", "pareto")

    def __init__(
        self,
        frequency_dist: str = "poisson",
        severity_dist: str = "lognormal",
        n_simulations: int = 100_000,
        confidence_level: float = 0.999,
        random_state: int | None = 42,
    ) -> None:
        if frequency_dist not in self.FREQ_DISTS:
            raise ValueError(f"frequency_dist must be one of {self.FREQ_DISTS}.")
        if severity_dist not in self.SEV_DISTS:
            raise ValueError(f"severity_dist must be one of {self.SEV_DISTS}.")
        self.frequency_dist = frequency_dist
        self.severity_dist = severity_dist
        self.n_simulations = n_simulations
        self.confidence_level = confidence_level
        self.random_state = random_state
        self._freq_params: dict = {}
        self._sev_params: dict = {}
        self._aggregate_losses: np.ndarray | None = None

    def fit(
        self,
        frequencies: pd.Series | np.ndarray,
        severities: pd.Series | np.ndarray,
    ) -> "LossDistributionApproach":
        """Fit frequency and severity distributions by MLE.

        Parameters
        ----------
        frequencies : array-like
            Annual or periodic loss event counts.
        severities : array-like
            Individual loss amounts (positive values).

        Returns
        -------
        self
        """
        freq = np.asarray(frequencies, dtype=float)
        sev = np.asarray(severities, dtype=float)

        # Fit frequency distribution
        if self.frequency_dist == "poisson":
            self._freq_params = {"mu": float(np.mean(freq))}
        elif self.frequency_dist == "negative_binomial":
            mu = np.mean(freq)
            var = np.var(freq, ddof=1)
            if var <= mu:
                var = mu + 1e-6
            r = mu**2 / (var - mu)
            p = r / (r + mu)
            self._freq_params = {"n": r, "p": p}

        # Fit severity distribution
        if self.severity_dist == "lognormal":
            log_sev = np.log(sev[sev > 0])
            self._sev_params = {"mu": float(np.mean(log_sev)), 
                                "sigma": float(np.std(log_sev, ddof=1))}
        elif self.severity_dist == "gamma":
            shape, loc, scale = stats.gamma.fit(sev, floc=0)
            self._sev_params = {"shape": shape, "scale": scale}
        elif self.severity_dist == "pareto":
            shape, loc, scale = stats.pareto.fit(sev, floc=0)
            self._sev_params = {"shape": shape, "scale": scale}

        return self

    def _sample_frequencies(self, rng: np.random.Generator, n: int) -> np.ndarray:
        if self.frequency_dist == "poisson":
            return rng.poisson(self._freq_params["mu"], size=n)
        else:
            return rng.negative_binomial(
                self._freq_params["n"], self._freq_params["p"], size=n
            )

    def _sample_severities(self, rng: np.random.Generator, n: int) -> np.ndarray:
        if self.severity_dist == "lognormal":
            return rng.lognormal(self._sev_params["mu"], self._sev_params["sigma"], size=n)
        elif self.severity_dist == "gamma":
            return rng.gamma(self._sev_params["shape"], self._sev_params["scale"], size=n)
        else:
            return stats.pareto.rvs(
                self._sev_params["shape"], scale=self._sev_params["scale"], size=n,
                random_state=int(rng.integers(0, 2**31)),
            )

    def simulate(self) -> np.ndarray:
        """Run Monte Carlo convolution to produce the aggregate loss distribution.

        Returns
        -------
        np.ndarray of shape (n_simulations,) — simulated annual losses.
        """
        if not self._freq_params:
            raise RuntimeError("Call fit() before simulate().")
        rng = np.random.default_rng(self.random_state)
        freq_samples = self._sample_frequencies(rng, self.n_simulations)
        total_losses = np.zeros(self.n_simulations)
        for i, n_events in enumerate(freq_samples):
            if n_events > 0:
                total_losses[i] = self._sample_severities(rng, int(n_events)).sum()
        self._aggregate_losses = total_losses
        return total_losses

    def capital_estimate(self) -> dict:
        """Compute regulatory capital as VaR and CVaR.

        Returns
        -------
        dict with keys: var_capital, cvar_capital, expected_loss, unexpected_loss.
        """
        if self._aggregate_losses is None:
            self.simulate()
        losses = self._aggregate_losses
        var = float(np.quantile(losses, self.confidence_level))
        cvar = float(np.mean(losses[losses >= var]))
        el = float(np.mean(losses))
        return {
            "expected_loss": el,
            "var_capital": var,
            "cvar_capital": cvar,
            "unexpected_loss": var - el,
            "confidence_level": self.confidence_level,
        }
