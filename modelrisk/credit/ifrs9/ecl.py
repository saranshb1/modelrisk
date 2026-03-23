"""ECL aggregation for IFRS 9.

ECL = PD × LGD × EAD × discount_factor

This module handles portfolio-level ECL computation from the outputs of
``LifetimePDCurve`` (or a simple 12-month PD for Stage 1).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class ECLCalculator:
    """Compute Expected Credit Losses at portfolio level.

    Handles both 12-month ECL (Stage 1) and lifetime ECL (Stage 2/3),
    and can aggregate across multiple scenario-weighted ECL inputs.

    Parameters
    ----------
    discount_rate : float
        Annual discount rate for discounting future ECL cash flows.
    period_type : str — ``'monthly'`` or ``'annual'``.

    Examples
    --------
    >>> ecl = ECLCalculator(discount_rate=0.05)
    >>> result = ecl.compute_portfolio(
    ...     pd_array=df['pd_12m'],
    ...     lgd_array=df['lgd'],
    ...     ead_array=df['ead'],
    ...     stage_array=df['stage'],
    ...     lifetime_pd_array=df['lifetime_pd'],
    ... )
    >>> ecl.summary(result)
    """

    def __init__(
        self,
        discount_rate: float = 0.05,
        period_type: str = "monthly",
    ) -> None:
        self.discount_rate = discount_rate
        self.period_type = period_type

    def compute_portfolio(
        self,
        pd_array: pd.Series | np.ndarray,
        lgd_array: pd.Series | np.ndarray,
        ead_array: pd.Series | np.ndarray,
        stage_array: pd.Series | np.ndarray,
        lifetime_pd_array: pd.Series | np.ndarray | None = None,
        remaining_periods_array: pd.Series | np.ndarray | None = None,
    ) -> pd.DataFrame:
        """Compute exposure-level ECL for a portfolio.

        Stage 1 exposures use the 12-month PD (``pd_array``).
        Stage 2 and 3 exposures use the lifetime PD (``lifetime_pd_array``).

        A simple single-period discount is applied: the ECL is assumed
        to crystallise at the midpoint of the exposure's life. For a
        full period-by-period calculation, use ``LifetimePDCurve``.

        Parameters
        ----------
        pd_array : array-like of shape (n)
            12-month point-in-time PD per exposure.
        lgd_array : array-like of shape (n)
            LGD per exposure (0 to 1).
        ead_array : array-like of shape (n)
            Exposure at default per exposure (monetary units).
        stage_array : array-like of shape (n)
            IFRS 9 stage (1, 2, or 3) per exposure.
        lifetime_pd_array : array-like of shape (n), optional
            Lifetime cumulative PD. Required for Stage 2/3 accuracy.
            If not provided, 12-month PD is used for all stages.
        remaining_periods_array : array-like of shape (n), optional
            Remaining life in periods — used to compute mid-life discount.

        Returns
        -------
        pd.DataFrame — one row per exposure with columns:
            pd_used, lgd, ead, stage, ecl, ecl_rate.
        """
        pd_arr = np.asarray(pd_array, dtype=float)
        lgd_arr = np.asarray(lgd_array, dtype=float)
        ead_arr = np.asarray(ead_array, dtype=float)
        stage_arr = np.asarray(stage_array, dtype=int)
        n = len(pd_arr)

        if lifetime_pd_array is not None:
            lifetime_pd_arr = np.asarray(lifetime_pd_array, dtype=float)
        else:
            lifetime_pd_arr = pd_arr.copy()

        # Select PD: Stage 1 → 12m PD, Stage 2/3 → lifetime PD
        pd_used = np.where(stage_arr == 1, pd_arr, lifetime_pd_arr)

        # Discount factor: mid-life approximation
        periods_per_year = 12 if self.period_type == "monthly" else 1
        r_period = self.discount_rate / periods_per_year
        if remaining_periods_array is not None:
            rem = np.asarray(remaining_periods_array, dtype=float)
            mid = rem / 2
        else:
            mid = np.ones(n) * periods_per_year / 2  # default: 6 months
        discount = 1.0 / (1.0 + r_period) ** mid

        ecl = pd_used * lgd_arr * ead_arr * discount

        return pd.DataFrame({
            "stage": stage_arr,
            "pd_12m": pd_arr,
            "lifetime_pd": lifetime_pd_arr,
            "pd_used": pd_used,
            "lgd": lgd_arr,
            "ead": ead_arr,
            "discount_factor": discount,
            "ecl": ecl,
            "ecl_rate": np.where(ead_arr > 0, ecl / ead_arr, 0.0),
        })

    def summary(self, portfolio_ecl: pd.DataFrame) -> pd.DataFrame:
        """Aggregate ECL summary by stage.

        Parameters
        ----------
        portfolio_ecl : pd.DataFrame — output of ``compute_portfolio()``.

        Returns
        -------
        pd.DataFrame — totals per stage plus grand total.
        """
        rows = []
        for stage in [1, 2, 3]:
            sub = portfolio_ecl[portfolio_ecl["stage"] == stage]
            rows.append({
                "stage": stage,
                "n_exposures": len(sub),
                "total_ead": sub["ead"].sum(),
                "total_ecl": sub["ecl"].sum(),
                "coverage_ratio": (
                    sub["ecl"].sum() / sub["ead"].sum()
                    if sub["ead"].sum() > 0 else 0.0
                ),
                "mean_pd_used": sub["pd_used"].mean() if len(sub) > 0 else 0.0,
            })
        total_ead = portfolio_ecl["ead"].sum()
        rows.append({
            "stage": "TOTAL",
            "n_exposures": len(portfolio_ecl),
            "total_ead": total_ead,
            "total_ecl": portfolio_ecl["ecl"].sum(),
            "coverage_ratio": (
                portfolio_ecl["ecl"].sum() / total_ead if total_ead > 0 else 0.0
            ),
            "mean_pd_used": portfolio_ecl["pd_used"].mean(),
        })
        return pd.DataFrame(rows)

    def weighted_ecl(
        self,
        scenario_ecls: dict[str, float],
        scenario_weights: dict[str, float],
    ) -> float:
        """Probability-weighted ECL across multiple scenarios.

        Parameters
        ----------
        scenario_ecls : dict — scenario name → total ECL (£/$ amount).
        scenario_weights : dict — scenario name → probability weight (must sum to 1.0).

        Returns
        -------
        float — probability-weighted total ECL.
        """
        total_weight = sum(scenario_weights.values())
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError(
                f"Scenario weights must sum to 1.0, got {total_weight:.4f}."
            )
        return float(sum(
            scenario_ecls[name] * scenario_weights[name]
            for name in scenario_ecls
        ))
