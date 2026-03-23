"""IFRS 9 Stage 1 / 2 / 3 classification.

Under IFRS 9, every financial instrument must be assigned to one of
three stages at each reporting date. Stage assignment drives whether
a 12-month or lifetime ECL is recognised.

Stage 1 — No significant increase in credit risk (SICR) since
           origination. 12-month ECL.
Stage 2 — SICR has occurred but no objective evidence of impairment.
           Lifetime ECL.
Stage 3 — Credit-impaired (defaulted). Lifetime ECL on impaired basis.

SICR detection supports three approaches:

``'absolute'``    — Stage 2 if current PD exceeds an absolute threshold
                    (e.g. > 1%). Simple and auditable.
``'relative'``    — Stage 2 if PD has increased by more than a relative
                    multiplier vs. origination PD (e.g. 2× or more).
                    More sensitive for low-PD portfolios.
``'dual'``        — Stage 2 if *either* absolute or relative trigger fires.
                    Basel-aligned best practice; recommended default.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class StagingClassifier:
    """Assign IFRS 9 stages to a portfolio of exposures.

    Parameters
    ----------
    method : str
        SICR detection method: ``'absolute'``, ``'relative'``, or ``'dual'``.
    absolute_threshold : float
        PD level above which Stage 2 is triggered (``'absolute'`` / ``'dual'``).
        Common values: 0.01 (1%) for retail, 0.005 for low-default portfolios.
    relative_multiplier : float
        PD multiple vs. origination above which Stage 2 fires
        (``'relative'`` / ``'dual'``). Common value: 2.0 (doubling of PD).
    default_threshold : float
        PD at or above which Stage 3 is assigned (objective impairment).
        Typically aligned with the IRB default definition (e.g. 20%).
    low_credit_risk_exemption : float or None
        If current PD is below this value, the exposure is always Stage 1
        regardless of relative change (the "low credit risk" exemption
        under IFRS 9 paragraph 5.5.10). Typical: 0.003 (30 bps).

    Examples
    --------
    >>> classifier = StagingClassifier(method='dual')
    >>> stages = classifier.classify(
    ...     current_pd=df['pd_current'],
    ...     origination_pd=df['pd_at_origination'],
    ...     is_defaulted=df['default_flag'],
    ... )
    >>> classifier.stage_summary(stages, current_pd=df['pd_current'])
    """

    METHODS = ("absolute", "relative", "dual")

    def __init__(
        self,
        method: str = "dual",
        absolute_threshold: float = 0.01,
        relative_multiplier: float = 2.0,
        default_threshold: float = 0.20,
        low_credit_risk_exemption: float | None = 0.003,
    ) -> None:
        if method not in self.METHODS:
            raise ValueError(f"method must be one of {self.METHODS}, got '{method}'.")
        self.method = method
        self.absolute_threshold = absolute_threshold
        self.relative_multiplier = relative_multiplier
        self.default_threshold = default_threshold
        self.low_credit_risk_exemption = low_credit_risk_exemption

    def classify(
        self,
        current_pd: pd.Series | np.ndarray,
        origination_pd: pd.Series | np.ndarray | None = None,
        is_defaulted: pd.Series | np.ndarray | None = None,
    ) -> np.ndarray:
        """Assign stages to each exposure.

        Parameters
        ----------
        current_pd : array-like of shape (n_exposures,)
            Current point-in-time PD for each exposure.
        origination_pd : array-like of shape (n_exposures,), optional
            PD at origination (or last annual review). Required when
            ``method`` is ``'relative'`` or ``'dual'``.
        is_defaulted : array-like of shape (n_exposures,), optional
            Boolean flag for known defaulted / credit-impaired exposures.
            If provided, these are assigned Stage 3 directly.

        Returns
        -------
        np.ndarray of shape (n_exposures,) with integer values 1, 2, or 3.
        """
        cur = np.asarray(current_pd, dtype=float)
        n = len(cur)
        stages = np.ones(n, dtype=int)  # default everyone to Stage 1

        # Stage 3 — defaulted
        if is_defaulted is not None:
            defaulted = np.asarray(is_defaulted, dtype=bool)
        else:
            defaulted = cur >= self.default_threshold
        stages[defaulted] = 3

        # SICR check for non-defaulted exposures
        non_def = ~defaulted

        # Low credit risk exemption — skip SICR test
        if self.low_credit_risk_exemption is not None:
            low_risk = cur < self.low_credit_risk_exemption
            non_def_not_low = non_def & ~low_risk
        else:
            non_def_not_low = non_def

        if self.method in ("absolute", "dual"):
            absolute_sicr = cur >= self.absolute_threshold
            stages[non_def_not_low & absolute_sicr] = 2

        if self.method in ("relative", "dual"):
            if origination_pd is None:
                raise ValueError(
                    f"origination_pd is required for method='{self.method}'."
                )
            orig = np.asarray(origination_pd, dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(orig > 0, cur / orig, np.inf)
            relative_sicr = ratio >= self.relative_multiplier
            stages[non_def_not_low & relative_sicr] = 2

        return stages

    def stage_summary(
        self,
        stages: np.ndarray,
        current_pd: pd.Series | np.ndarray | None = None,
        exposure_at_default: pd.Series | np.ndarray | None = None,
    ) -> pd.DataFrame:
        """Portfolio-level stage summary.

        Parameters
        ----------
        stages : np.ndarray of stage assignments (1, 2, or 3).
        current_pd : array-like, optional — used to compute mean PD per stage.
        exposure_at_default : array-like, optional — EAD for exposure-weighted summary.

        Returns
        -------
        pd.DataFrame with columns: stage, count, pct_count,
            mean_pd (if current_pd provided), total_ead (if ead provided).
        """
        rows = []
        for s in [1, 2, 3]:
            mask = stages == s
            count = int(mask.sum())
            row: dict = {
                "stage": s,
                "count": count,
                "pct_count": round(count / len(stages) * 100, 2),
            }
            if current_pd is not None:
                pd_arr = np.asarray(current_pd, dtype=float)
                row["mean_pd"] = round(float(pd_arr[mask].mean()), 6) if count > 0 else 0.0
            if exposure_at_default is not None:
                ead_arr = np.asarray(exposure_at_default, dtype=float)
                row["total_ead"] = float(ead_arr[mask].sum())
                row["pct_ead"] = round(
                    ead_arr[mask].sum() / ead_arr.sum() * 100, 2
                ) if ead_arr.sum() > 0 else 0.0
            rows.append(row)
        return pd.DataFrame(rows)
