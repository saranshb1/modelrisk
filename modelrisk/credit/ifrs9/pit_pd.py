"""Point-in-time (PIT) PD calibration for IFRS 9.

PIT calibration adjusts raw model scores so that predicted default
rates reflect *current* economic conditions rather than long-run
averages. Two approaches are implemented:

1. **Scalar adjustment** — multiply raw PDs by the ratio of the
   recent observed default rate to the model's long-run average.
   Simple, transparent, audit-friendly.

2. **Exponential time weighting** — weight training observations
   by recency before re-fitting or recalibrating the model's
   output layer, giving recent default experience more influence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class PITCalibrator:
    """Calibrate raw PD scores to point-in-time default rates.

    Wraps a fitted ``BasePDModel`` and applies a calibration layer so
    that predicted probabilities match recent observed default rates.

    Provides two calibration methods selectable via ``method``:

    ``'scalar'``
        Multiplies raw PDs by ``recent_dr / model_long_run_dr``.
        Capped at 1.0. Fast and explainable — preferred for
        regulatory submissions where simplicity is valued.

    ``'isotonic'``
        Fits isotonic regression mapping raw scores to observed
        default rates, monotonically preserving score ranking.
        Better when the score distribution has shifted significantly.

    ``'platt'``
        Fits a logistic regression (Platt scaling) on top of raw
        scores to recalibrate probabilities. Flexible but requires
        a held-out calibration set.

    Parameters
    ----------
    method : str
        Calibration method: ``'scalar'``, ``'isotonic'``, or ``'platt'``.
    halflife_months : int
        Half-life for exponential time weighting (used when
        ``fit_with_time_weights=True`` in ``calibrate()``).
        Observations this many months old receive weight 0.5.
        Shorter = more responsive to recent data.
        IFRS 9 typical range: 12–24 months.
    min_pd : float
        Floor applied to all calibrated PDs. Prevents zero PD.
    max_pd : float
        Cap applied to all calibrated PDs.

    Examples
    --------
    >>> cal = PITCalibrator(method='scalar', halflife_months=18)
    >>> cal.calibrate(raw_pd=model.predict_proba(X),
    ...               observed_default_rate=0.032,
    ...               model_long_run_dr=0.018)
    >>> cal.transform(model.predict_proba(X_new))
    """

    METHODS = ("scalar", "isotonic", "platt")

    def __init__(
        self,
        method: str = "scalar",
        halflife_months: int = 18,
        min_pd: float = 0.0001,
        max_pd: float = 0.9999,
    ) -> None:
        if method not in self.METHODS:
            raise ValueError(f"method must be one of {self.METHODS}, got '{method}'.")
        self.method = method
        self.halflife_months = halflife_months
        self.min_pd = min_pd
        self.max_pd = max_pd

        self._scalar: float | None = None
        self._isotonic: IsotonicRegression | None = None
        self._platt: LogisticRegression | None = None
        self._long_run_dr: float | None = None

    # ------------------------------------------------------------------
    # Time weighting utility
    # ------------------------------------------------------------------

    @staticmethod
    def exponential_weights(
        observation_dates: pd.Series,
        reference_date: pd.Timestamp | None = None,
        halflife_months: int = 18,
    ) -> np.ndarray:
        """Compute exponential time decay weights.

        More recent observations receive higher weight. Used to
        bias model training or calibration towards current conditions.

        Parameters
        ----------
        observation_dates : pd.Series of datetime-like values.
            Date each observation was recorded.
        reference_date : pd.Timestamp or None
            The "now" anchor. Defaults to the most recent date in the series.
        halflife_months : int
            Months after which weight halves.

        Returns
        -------
        np.ndarray of shape (n_samples,) — weights summing to n_samples.
        """
        dates = pd.to_datetime(observation_dates)
        ref = reference_date or dates.max()
        age_months = (ref - dates).dt.days / 30.44
        decay = np.exp(-np.log(2) * age_months / halflife_months)
        # Normalise so weights sum to n_samples (preserves scale)
        return decay / decay.mean()

    # ------------------------------------------------------------------
    # Calibration fit
    # ------------------------------------------------------------------

    def calibrate(
        self,
        raw_pd: np.ndarray,
        observed_default_rate: float | None = None,
        model_long_run_dr: float | None = None,
        y_cal: np.ndarray | None = None,
    ) -> PITCalibrator:
        """Fit the calibration layer.

        Parameters
        ----------
        raw_pd : np.ndarray of shape (n_samples,)
            Raw model output probabilities (uncalibrated).
        observed_default_rate : float, optional
            Recent observed default rate (required for ``'scalar'``).
        model_long_run_dr : float, optional
            Model's long-run average PD (required for ``'scalar'``).
            Typically the mean of ``raw_pd`` on the full development sample.
        y_cal : np.ndarray of shape (n_samples,), optional
            Actual binary defaults (required for ``'isotonic'``
            and ``'platt'``).

        Returns
        -------
        self
        """
        raw_pd = np.asarray(raw_pd, dtype=float)

        if self.method == "scalar":
            if observed_default_rate is None or model_long_run_dr is None:
                raise ValueError(
                    "observed_default_rate and model_long_run_dr are required "
                    "for method='scalar'."
                )
            self._scalar = observed_default_rate / max(model_long_run_dr, 1e-9)
            self._long_run_dr = model_long_run_dr

        elif self.method == "isotonic":
            if y_cal is None:
                raise ValueError("y_cal is required for method='isotonic'.")
            self._isotonic = IsotonicRegression(out_of_bounds="clip")
            self._isotonic.fit(raw_pd, np.asarray(y_cal, dtype=float))

        elif self.method == "platt":
            if y_cal is None:
                raise ValueError("y_cal is required for method='platt'.")
            self._platt = LogisticRegression(C=1e9, solver="lbfgs", max_iter=1000)
            self._platt.fit(raw_pd.reshape(-1, 1), np.asarray(y_cal, dtype=int))

        return self

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def transform(self, raw_pd: np.ndarray) -> np.ndarray:
        """Apply calibration to new raw PD scores.

        Parameters
        ----------
        raw_pd : np.ndarray of shape (n_samples,)

        Returns
        -------
        np.ndarray of shape (n_samples,) — calibrated PIT PDs.
        """
        raw_pd = np.asarray(raw_pd, dtype=float)

        if self.method == "scalar":
            if self._scalar is None:
                raise RuntimeError("Call calibrate() first.")
            calibrated = raw_pd * self._scalar

        elif self.method == "isotonic":
            if self._isotonic is None:
                raise RuntimeError("Call calibrate() first.")
            calibrated = self._isotonic.predict(raw_pd)

        elif self.method == "platt":
            if self._platt is None:
                raise RuntimeError("Call calibrate() first.")
            calibrated = self._platt.predict_proba(raw_pd.reshape(-1, 1))[:, 1]

        return np.clip(calibrated, self.min_pd, self.max_pd)

    def calibration_summary(self) -> pd.Series:
        """Return a summary of the calibration parameters."""
        if self.method == "scalar":
            if self._scalar is None:
                raise RuntimeError("Not calibrated yet.")
            return pd.Series({
                "method": self.method,
                "scalar_multiplier": round(self._scalar, 6),
                "long_run_dr": self._long_run_dr,
                "halflife_months": self.halflife_months,
                "min_pd": self.min_pd,
                "max_pd": self.max_pd,
            })
        return pd.Series({
            "method": self.method,
            "halflife_months": self.halflife_months,
            "min_pd": self.min_pd,
            "max_pd": self.max_pd,
        })
