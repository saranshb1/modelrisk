"""Forward marginal PD curves for IFRS 9 lifetime ECL."""

from __future__ import annotations

import numpy as np
import pandas as pd


class ForwardPDCurve:
    """Build term-structure of marginal conditional PDs.

    Converts a single 12-month PIT PD into a multi-period term structure
    of marginal default probabilities — the probability of defaulting in
    period t given survival to period t-1.

    Two projection methods are supported:

    ``'constant_hazard'``
        Assumes a constant monthly hazard rate derived from the 12-month PD.
        Simple and internally consistent. Best for stable portfolios.

    ``'seasoning_curve'``
        Applies a seasoning multiplier per period, reflecting that newly
        originated loans often have different default timing to seasoned ones.
        Requires a ``seasoning_factors`` array (one per period).

    Parameters
    ----------
    n_periods : int
        Number of forward periods (months by default).
    period_type : str
        ``'monthly'`` or ``'annual'``.

    Examples
    --------
    >>> curve = ForwardPDCurve(n_periods=60)
    >>> marginal_pds = curve.build(pit_pd_12m=0.03)
    >>> curve.cumulative_pd()
    """

    def __init__(
        self,
        n_periods: int = 60,
        period_type: str = "monthly",
    ) -> None:
        if period_type not in ("monthly", "annual"):
            raise ValueError("period_type must be 'monthly' or 'annual'.")
        self.n_periods = n_periods
        self.period_type = period_type
        self._marginal_pd: np.ndarray | None = None

    def build(
        self,
        pit_pd_12m: float,
        method: str = "constant_hazard",
        seasoning_factors: np.ndarray | None = None,
    ) -> np.ndarray:
        """Build marginal PD term structure from a 12-month PIT PD.

        Parameters
        ----------
        pit_pd_12m : float
            12-month point-in-time PD (e.g. 0.03 for 3%).
        method : str
            ``'constant_hazard'`` or ``'seasoning_curve'``.
        seasoning_factors : np.ndarray, optional
            Multipliers per period (length = n_periods). Required for
            ``'seasoning_curve'``. Values around 1.0 leave PD unchanged;
            > 1 increases it; < 1 decreases it.

        Returns
        -------
        np.ndarray of shape (n_periods,) — marginal PD per period.
        """
        if method == "constant_hazard":
            # Monthly hazard from annual PD: h = -ln(1 - PD) / 12
            periods_per_year = 12 if self.period_type == "monthly" else 1
            annual_pd = pit_pd_12m
            hazard = -np.log(max(1 - annual_pd, 1e-9)) / periods_per_year
            survival = np.exp(-hazard * np.arange(1, self.n_periods + 1))
            survival_prev = np.concatenate([[1.0], survival[:-1]])
            marginal = survival_prev - survival

        elif method == "seasoning_curve":
            if seasoning_factors is None:
                raise ValueError("seasoning_factors required for 'seasoning_curve'.")
            factors = np.asarray(seasoning_factors, dtype=float)
            if len(factors) != self.n_periods:
                raise ValueError(
                    f"seasoning_factors must have length {self.n_periods}, "
                    f"got {len(factors)}."
                )
            periods_per_year = 12 if self.period_type == "monthly" else 1
            base_hazard = -np.log(max(1 - pit_pd_12m, 1e-9)) / periods_per_year
            hazards = base_hazard * factors
            survival = np.cumprod(np.exp(-hazards))
            survival_prev = np.concatenate([[1.0], survival[:-1]])
            marginal = survival_prev - survival

        else:
            raise ValueError(f"Unknown method '{method}'.")

        self._marginal_pd = np.clip(marginal, 0.0, 1.0)
        return self._marginal_pd

    def cumulative_pd(self) -> np.ndarray:
        """Return cumulative PD (1 - survival) at each period.

        Returns
        -------
        np.ndarray of shape (n_periods,).
        """
        if self._marginal_pd is None:
            raise RuntimeError("Call build() first.")
        survival = 1.0 - np.cumsum(self._marginal_pd)
        return np.clip(1.0 - survival, 0.0, 1.0)

    def as_dataframe(self) -> pd.DataFrame:
        """Return the term structure as a tidy DataFrame."""
        if self._marginal_pd is None:
            raise RuntimeError("Call build() first.")
        return pd.DataFrame({
            "period": np.arange(1, self.n_periods + 1),
            "marginal_pd": self._marginal_pd,
            "cumulative_pd": self.cumulative_pd(),
            "survival": 1.0 - self.cumulative_pd(),
        })
