"""Through-the-cycle (TTC) PD calibration for Basel IRB."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


class TTCCalibrator:
    """Calibrate PD estimates to through-the-cycle long-run averages.

    IRB regulations (Basel II/III, CRR Art. 180) require that PD estimates
    reflect the long-run average default rate observed over a complete
    economic cycle — typically a minimum of 5 years, preferably 7 or more.

    Parameters
    ----------
    min_pd : float
        Regulatory floor on TTC PD. Basel typically requires min 0.03%
        (0.0003) for non-defaulted exposures.
    cycle_length_years : int
        Expected length of a full credit cycle. Used for diagnostics.

    Examples
    --------
    >>> cal = TTCCalibrator(min_pd=0.0003)
    >>> cal.fit(annual_default_rates)
    >>> cal.long_run_average_pd
    >>> cal.apply(segment_pit_pd=0.025)
    """

    def __init__(
        self,
        min_pd: float = 0.0003,
        cycle_length_years: int = 7,
    ) -> None:
        self.min_pd = min_pd
        self.cycle_length_years = cycle_length_years
        self._annual_drs: np.ndarray | None = None
        self._long_run_avg: float | None = None
        self._std: float | None = None

    def fit(
        self,
        annual_default_rates: pd.Series | np.ndarray,
        weights: np.ndarray | None = None,
    ) -> TTCCalibrator:
        """Estimate long-run average default rate from historical data.

        Parameters
        ----------
        annual_default_rates : array-like
            Observed annual default rates per year (e.g. [0.01, 0.02, 0.015...]).
            Should span at least one full credit cycle.
        weights : array-like or None
            Optional weights per year (e.g. exposure-weighted). If None,
            simple unweighted average is used.

        Returns
        -------
        self
        """
        drs = np.asarray(annual_default_rates, dtype=float)
        self._annual_drs = drs
        if weights is not None:
            w = np.asarray(weights, dtype=float)
            self._long_run_avg = float(np.average(drs, weights=w))
        else:
            self._long_run_avg = float(np.mean(drs))
        self._std = float(np.std(drs, ddof=1))
        return self

    @property
    def long_run_average_pd(self) -> float:
        """Long-run average PD floored at min_pd."""
        if self._long_run_avg is None:
            raise RuntimeError("Call fit() first.")
        return max(self._long_run_avg, self.min_pd)

    def apply(self, segment_pit_pd: float | np.ndarray) -> float | np.ndarray:
        """Derive TTC PD from a PIT PD estimate.

        Scales PIT PD proportionally so the portfolio-level TTC PD
        equals the long-run average. Individual exposure rankings are
        preserved.

        Parameters
        ----------
        segment_pit_pd : float or array-like
            PIT PD(s) for a segment or individual exposure.

        Returns
        -------
        float or np.ndarray — TTC PD(s), floored at min_pd.
        """
        if self._long_run_avg is None:
            raise RuntimeError("Call fit() first.")
        scalar = self.long_run_average_pd / max(
            float(np.mean(np.asarray(segment_pit_pd, dtype=float))), 1e-9
        )
        ttc = np.asarray(segment_pit_pd, dtype=float) * scalar
        ttc = np.clip(ttc, self.min_pd, 0.9999)
        return float(ttc) if np.isscalar(segment_pit_pd) else ttc

    def calibration_summary(self) -> pd.Series:
        """Return calibration diagnostics."""
        if self._annual_drs is None:
            raise RuntimeError("Call fit() first.")
        n = len(self._annual_drs)
        return pd.Series({
            "n_years_data": n,
            "long_run_avg_pd": self.long_run_average_pd,
            "std_annual_dr": self._std,
            "min_annual_dr": float(self._annual_drs.min()),
            "max_annual_dr": float(self._annual_drs.max()),
            "cycle_peak_trough_ratio": (
                float(self._annual_drs.max() / max(self._annual_drs.min(), 1e-9))
            ),
            "min_pd_floor": self.min_pd,
            "data_covers_full_cycle": n >= self.cycle_length_years,
        })
