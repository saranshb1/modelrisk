"""Lifetime PD survival curves for IFRS 9 Stage 2 and Stage 3."""

from __future__ import annotations

import numpy as np
import pandas as pd


class LifetimePDCurve:
    """Construct lifetime PD curves from forward marginal PDs.

    Combines per-exposure forward PD term structures (from
    ``ForwardPDCurve``) with remaining contractual maturity to produce
    the exposure-specific lifetime cumulative PD needed for Stage 2 ECL.

    Parameters
    ----------
    discount_rate : float
        Annual discount rate for computing present values (e.g. 0.05).
        Used when computing discounted lifetime PD for ECL.
    period_type : str
        ``'monthly'`` or ``'annual'`` — must match the ForwardPDCurve.

    Examples
    --------
    >>> curve = LifetimePDCurve(discount_rate=0.05)
    >>> lifetime_pd = curve.compute(marginal_pds, remaining_periods=48)
    >>> curve.discounted_ecl_weights(lgd=0.45, ead=100_000)
    """

    def __init__(
        self,
        discount_rate: float = 0.05,
        period_type: str = "monthly",
    ) -> None:
        self.discount_rate = discount_rate
        self.period_type = period_type
        self._marginal_pd: np.ndarray | None = None
        self._remaining_periods: int | None = None

    def compute(
        self,
        marginal_pds: np.ndarray,
        remaining_periods: int,
    ) -> "LifetimePDCurve":
        """Store the forward PD curve truncated to remaining maturity.

        Parameters
        ----------
        marginal_pds : np.ndarray of shape (n_periods,)
            Full forward marginal PD term structure.
        remaining_periods : int
            Remaining contractual life in periods. Must be <= len(marginal_pds).

        Returns
        -------
        self
        """
        pds = np.asarray(marginal_pds, dtype=float)
        if remaining_periods > len(pds):
            raise ValueError(
                f"remaining_periods ({remaining_periods}) exceeds "
                f"len(marginal_pds) ({len(pds)})."
            )
        self._marginal_pd = pds[:remaining_periods]
        self._remaining_periods = remaining_periods
        return self

    @property
    def lifetime_pd(self) -> float:
        """Total cumulative default probability over remaining life."""
        if self._marginal_pd is None:
            raise RuntimeError("Call compute() first.")
        survival = np.cumprod(1.0 - self._marginal_pd)
        return float(1.0 - survival[-1])

    def discount_factors(self) -> np.ndarray:
        """Compute per-period discount factors.

        Returns
        -------
        np.ndarray of shape (remaining_periods,).
        """
        if self._remaining_periods is None:
            raise RuntimeError("Call compute() first.")
        periods_per_year = 12 if self.period_type == "monthly" else 1
        r_period = self.discount_rate / periods_per_year
        t = np.arange(1, self._remaining_periods + 1)
        return 1.0 / (1.0 + r_period) ** t

    def discounted_ecl_weights(
        self, lgd: float, ead: float
    ) -> pd.DataFrame:
        """Compute period-by-period discounted ECL contributions.

        ECL_t = marginal_PD_t × LGD × EAD_t × discount_factor_t

        This assumes flat EAD across periods (a simplification).
        For amortising products, pass a declining EAD schedule instead.

        Parameters
        ----------
        lgd : float — loss given default (0 to 1).
        ead : float — exposure at default (current balance).

        Returns
        -------
        pd.DataFrame with columns: period, marginal_pd, discount_factor,
            ecl_contribution, cumulative_ecl.
        """
        if self._marginal_pd is None:
            raise RuntimeError("Call compute() first.")
        df_factors = self.discount_factors()
        ecl_contrib = self._marginal_pd * lgd * ead * df_factors
        return pd.DataFrame({
            "period": np.arange(1, self._remaining_periods + 1),
            "marginal_pd": self._marginal_pd,
            "discount_factor": df_factors,
            "ecl_contribution": ecl_contrib,
            "cumulative_ecl": np.cumsum(ecl_contrib),
        })

    def total_ecl(self, lgd: float, ead: float) -> float:
        """Total lifetime ECL for a single exposure.

        Parameters
        ----------
        lgd : float
        ead : float

        Returns
        -------
        float — discounted lifetime ECL.
        """
        return float(self.discounted_ecl_weights(lgd, ead)["ecl_contribution"].sum())
