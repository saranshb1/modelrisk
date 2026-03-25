"""IRB cycle adjuster — Pluto-Tasche and scalar smoothing."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize, stats


class CycleAdjuster:
    """Smooth PIT default rates to TTC using cycle-adjustment methods.

    Methods
    -------
    ``'scalar'``      Multiply PIT PD by long-run / recent ratio.
    ``'hp_filter'``   Hodrick-Prescott filter to extract trend component.
    ``'moving_avg'``  Rolling average over a configurable window.

    Examples
    --------
    >>> adj = CycleAdjuster(method='hp_filter', hp_lambda=1600)
    >>> ttc_series = adj.smooth(pit_default_rate_series)
    """

    def __init__(
        self,
        method: str = "moving_avg",
        window: int = 5,
        hp_lambda: float = 1600,
    ) -> None:
        if method not in ("scalar", "hp_filter", "moving_avg"):
            raise ValueError("method must be 'scalar', 'hp_filter', or 'moving_avg'.")
        self.method = method
        self.window = window
        self.hp_lambda = hp_lambda

    def smooth(self, pit_series: pd.Series | np.ndarray) -> np.ndarray:
        """Smooth a time series of PIT default rates.

        Parameters
        ----------
        pit_series : array-like of annual default rates.

        Returns
        -------
        np.ndarray — smoothed TTC default rate series.
        """
        x = np.asarray(pit_series, dtype=float)

        if self.method == "scalar":
            long_run = x.mean()
            return np.full_like(x, long_run)

        elif self.method == "moving_avg":
            series = pd.Series(x)
            smoothed = series.rolling(self.window, min_periods=1, center=True).mean()
            return smoothed.values

        elif self.method == "hp_filter":
            return self._hp_filter(x)

        return x

    def _hp_filter(self, y: np.ndarray) -> np.ndarray:
        """Hodrick-Prescott filter trend extraction."""
        n = len(y)
        lam = self.hp_lambda
        # Build second-difference matrix
        diff2 = np.zeros((n - 2, n))
        for i in range(n - 2):
            diff2[i, i] = 1
            diff2[i, i + 1] = -2
            diff2[i, i + 2] = 1
        eye = np.eye(n)
        trend = np.linalg.solve(eye + lam * diff2.T @ diff2, y)
        return trend


class RatingMasterScale:
    """Map continuous PD estimates to a discrete rating master scale.

    Provides a configurable master scale with named grades (e.g. 1–25),
    each defined by a PD band. Useful for IRB model outputs where
    regulators require PD estimates to be expressed as grade-level
    long-run averages rather than individual exposure estimates.

    Parameters
    ----------
    n_grades : int
        Number of rating grades (typically 15–25 for corporate/retail IRB).
    scale : pd.DataFrame or None
        Custom master scale with columns: grade, pd_lower, pd_upper, ttc_pd.
        If None, a logarithmically-spaced default scale is created.

    Examples
    --------
    >>> rms = RatingMasterScale(n_grades=18)
    >>> grades = rms.assign_grades(ttc_pd_array)
    >>> rms.scale_table()
    """

    def __init__(
        self,
        n_grades: int = 18,
        scale: pd.DataFrame | None = None,
    ) -> None:
        self.n_grades = n_grades
        if scale is not None:
            self._scale = scale
        else:
            self._scale = self._build_default_scale(n_grades)

    @staticmethod
    def _build_default_scale(n: int) -> pd.DataFrame:
        """Build a log-spaced master scale from 0.01% to 99%."""
        pd_upper = np.logspace(np.log10(0.0001), np.log10(0.99), n)
        pd_lower = np.concatenate([[0.0], pd_upper[:-1]])
        ttc_pd = (pd_lower + pd_upper) / 2
        ttc_pd[0] = pd_upper[0] / 2
        return pd.DataFrame({
            "grade": np.arange(1, n + 1),
            "pd_lower": pd_lower,
            "pd_upper": pd_upper,
            "ttc_pd": ttc_pd,
        })

    def assign_grades(self, ttc_pd: pd.Series | np.ndarray) -> np.ndarray:
        """Map TTC PD values to rating grades.

        Parameters
        ----------
        ttc_pd : array-like of TTC PD estimates.

        Returns
        -------
        np.ndarray of integer grade assignments.
        """
        pd_arr = np.asarray(ttc_pd, dtype=float)
        grades = np.full(len(pd_arr), self.n_grades, dtype=int)
        for _, row in self._scale.iterrows():
            mask = (pd_arr >= row["pd_lower"]) & (pd_arr < row["pd_upper"])
            grades[mask] = int(row["grade"])
        return grades

    def grade_pd(self, grade: int) -> float:
        """Return the representative TTC PD for a grade."""
        row = self._scale[self._scale["grade"] == grade]
        if row.empty:
            raise ValueError(f"Grade {grade} not in master scale.")
        return float(row["ttc_pd"].iloc[0])

    def scale_table(self) -> pd.DataFrame:
        """Return the full master scale as a formatted DataFrame."""
        df = self._scale.copy()
        df["pd_lower_pct"] = (df["pd_lower"] * 100).round(4)
        df["pd_upper_pct"] = (df["pd_upper"] * 100).round(4)
        df["ttc_pd_pct"] = (df["ttc_pd"] * 100).round(4)
        return df[["grade", "pd_lower_pct", "pd_upper_pct", "ttc_pd_pct"]]


class PITtoTTCBridge:
    """Convert PIT PD estimates to TTC for IRB reporting.

    Formalises the common bank practice of building one statistical
    model and deriving both IFRS 9 (PIT) and IRB (TTC) PDs from it,
    rather than maintaining two separate development processes.

    Parameters
    ----------
    method : str
        ``'scalar'`` — multiply PIT by long-run / current ratio.
        ``'logit_offset'`` — shift in log-odds space (more stable for
        extreme PDs).

    Examples
    --------
    >>> bridge = PITtoTTCBridge(method='scalar')
    >>> bridge.fit(historical_pit_pds, historical_ttc_pds)
    >>> ttc_pds = bridge.convert(new_pit_pds)
    """

    def __init__(self, method: str = "scalar") -> None:
        if method not in ("scalar", "logit_offset"):
            raise ValueError("method must be 'scalar' or 'logit_offset'.")
        self.method = method
        self._scalar: float | None = None
        self._logit_offset: float | None = None

    def fit(
        self,
        pit_pds: pd.Series | np.ndarray,
        ttc_pds: pd.Series | np.ndarray,
    ) -> "PITtoTTCBridge":
        """Estimate the PIT → TTC conversion from historical paired data.

        Parameters
        ----------
        pit_pds : array-like — historical PIT PD estimates.
        ttc_pds : array-like — corresponding TTC PD estimates.

        Returns
        -------
        self
        """
        pit = np.asarray(pit_pds, dtype=float)
        ttc = np.asarray(ttc_pds, dtype=float)

        if self.method == "scalar":
            self._scalar = float(np.mean(ttc) / max(np.mean(pit), 1e-9))
        else:
            logit = lambda p: np.log(np.clip(p, 1e-9, 1 - 1e-9) /
                                     (1 - np.clip(p, 1e-9, 1 - 1e-9)))
            self._logit_offset = float(np.mean(logit(ttc) - logit(pit)))
        return self

    def convert(self, pit_pds: pd.Series | np.ndarray) -> np.ndarray:
        """Convert PIT PDs to TTC PDs.

        Parameters
        ----------
        pit_pds : array-like of PIT PD estimates.

        Returns
        -------
        np.ndarray of TTC PD estimates.
        """
        pit = np.asarray(pit_pds, dtype=float)

        if self.method == "scalar":
            if self._scalar is None:
                raise RuntimeError("Call fit() first.")
            ttc = pit * self._scalar
        else:
            if self._logit_offset is None:
                raise RuntimeError("Call fit() first.")
            logit_pit = np.log(np.clip(pit, 1e-9, 1 - 1e-9) / (1 - np.clip(pit, 1e-9, 1 - 1e-9)))
            ttc = 1.0 / (1.0 + np.exp(-(logit_pit + self._logit_offset)))

        return np.clip(ttc, 0.0003, 0.9999)


class IRBCapital:
    """Basel IRB risk-weight formula for retail and corporate exposures.

    Implements the asymptotic single risk factor (ASRF) model underlying
    the Basel II/III/IV standardised IRB formula. Computes risk-weighted
    assets (RWA) from TTC PD, LGD, EAD, and maturity.

    References
    ----------
    Basel Committee on Banking Supervision (2006). Basel II: International
    Convergence of Capital Measurement and Capital Standards. Paragraphs
    272–276 (corporate) and 328–330 (retail).

    Examples
    --------
    >>> irb = IRBCapital(asset_class='retail_mortgage')
    >>> rwa = irb.compute_rwa(pd=0.005, lgd=0.15, ead=200_000)
    >>> irb.rwa_portfolio(df['ttc_pd'], df['lgd'], df['ead'])
    """

    ASSET_CLASSES = ("corporate", "retail_mortgage", "retail_other", "sme_retail")

    # Basel correlation parameters per asset class
    _CORRELATION = {
        "corporate":       (0.12, 0.24),   # R_min, R_max
        "retail_mortgage": (0.15, 0.15),   # fixed
        "retail_other":    (0.03, 0.16),
        "sme_retail":      (0.03, 0.16),
    }

    def __init__(
        self,
        asset_class: str = "corporate",
        confidence_level: float = 0.999,
        maturity_adjustment: bool = True,
    ) -> None:
        if asset_class not in self.ASSET_CLASSES:
            raise ValueError(f"asset_class must be one of {self.ASSET_CLASSES}.")
        self.asset_class = asset_class
        self.confidence_level = confidence_level
        self.maturity_adjustment = maturity_adjustment

    def _correlation(self, pd: float | np.ndarray) -> float | np.ndarray:
        r_min, r_max = self._CORRELATION[self.asset_class]
        if r_min == r_max:
            return r_min
        k = 50
        e1 = np.exp(-k * pd)
        e2 = np.exp(-k)
        return r_min * (1 - e1) / (1 - e2) + r_max * (1 - (1 - e1) / (1 - e2))

    def _maturity_adj(self, pd: float | np.ndarray, maturity: float) -> float | np.ndarray:
        """Basel maturity adjustment factor b(PD)."""
        b = (0.11852 - 0.05478 * np.log(np.clip(pd, 1e-9, 1.0))) ** 2
        return (1 + (maturity - 2.5) * b) / (1 - 1.5 * b)

    def compute_rwa(
        self,
        pd: float,
        lgd: float,
        ead: float,
        maturity: float = 2.5,
    ) -> dict:
        """Compute RWA for a single exposure.

        Parameters
        ----------
        pd : float — TTC PD (0 to 1).
        lgd : float — LGD (0 to 1).
        ead : float — EAD in monetary units.
        maturity : float — effective maturity in years (corporate only).

        Returns
        -------
        dict with keys: pd, lgd, ead, correlation, capital_requirement,
            rwa, expected_loss.
        """
        pd_c = max(float(pd), 0.0003)
        corr_r = self._correlation(pd_c)
        z = stats.norm.ppf(self.confidence_level)
        z_pd = stats.norm.ppf(pd_c)

        conditional_pd = stats.norm.cdf(
            (z_pd + np.sqrt(corr_r) * z) / np.sqrt(1 - corr_r)
        )
        capital_k = lgd * conditional_pd - lgd * pd_c

        if self.maturity_adjustment and self.asset_class == "corporate":
            capital_k *= self._maturity_adj(pd_c, maturity)

        capital_k = max(capital_k, 0.0)
        rwa = capital_k * 12.5 * ead
        el = pd_c * lgd * ead

        return {
            "pd": pd_c, "lgd": lgd, "ead": ead,
            "correlation": float(corr_r),
            "capital_requirement": float(capital_k),
            "rwa": float(rwa),
            "expected_loss": float(el),
        }

    def rwa_portfolio(
        self,
        pd_array: pd.Series | np.ndarray,
        lgd_array: pd.Series | np.ndarray,
        ead_array: pd.Series | np.ndarray,
        maturity_array: pd.Series | np.ndarray | None = None,
    ) -> pd.DataFrame:
        """Compute RWA for a portfolio of exposures.

        Returns
        -------
        pd.DataFrame with per-exposure RWA and portfolio totals row.
        """
        pds = np.asarray(pd_array, dtype=float)
        lgds = np.asarray(lgd_array, dtype=float)
        eads = np.asarray(ead_array, dtype=float)
        mats = (
            np.asarray(maturity_array, dtype=float)
            if maturity_array is not None
            else np.full(len(pds), 2.5)
        )
        rows = [self.compute_rwa(p, l, e, m) for p, l, e, m in zip(pds, lgds, eads, mats)]
        df = pd.DataFrame(rows)
        total = pd.DataFrame([{
            "pd": np.nan, "lgd": np.nan,
            "ead": df["ead"].sum(),
            "correlation": np.nan,
            "capital_requirement": np.nan,
            "rwa": df["rwa"].sum(),
            "expected_loss": df["expected_loss"].sum(),
        }])
        total.index = ["TOTAL"]
        return pd.concat([df, total])


class IRBValidator:
    """Basel IRB backtesting — traffic light tests and binomial tests.

    Validates whether a TTC PD model is performing within acceptable
    bounds by comparing predicted PDs against realised default rates
    over a backtesting window.

    References
    ----------
    EBA Guidelines on PD estimation, LGD estimation and treatment of
    defaulted assets (EBA/GL/2017/16), Section 5.

    Examples
    --------
    >>> val = IRBValidator()
    >>> val.traffic_light_test(predicted_pd=0.008, observed_dr=0.015, n_obligors=500)
    >>> val.binomial_test(predicted_pd=0.008, observed_dr=0.015, n_obligors=500)
    """

    TRAFFIC_LIGHT_ZONES = {
        "green":  "Model performing within tolerance",
        "amber":  "Elevated deviation — review calibration",
        "red":    "Significant deviation — recalibration required",
    }

    def traffic_light_test(
        self,
        predicted_pd: float,
        observed_dr: float,
        n_obligors: int,
        green_threshold: float = 0.20,
        amber_threshold: float = 0.50,
    ) -> dict:
        """Basel traffic light test on a single rating grade or segment.

        Compares relative deviation of observed default rate from predicted PD.

        Parameters
        ----------
        predicted_pd : float — model TTC PD for the grade/segment.
        observed_dr : float — realised default rate in the backtesting year.
        n_obligors : int — number of obligors in the grade/segment.
        green_threshold : float — max acceptable relative deviation for green.
        amber_threshold : float — max acceptable relative deviation for amber.

        Returns
        -------
        dict with keys: zone, relative_deviation, predicted_pd, observed_dr,
            n_obligors, interpretation.
        """
        if predicted_pd <= 0:
            raise ValueError("predicted_pd must be positive.")
        rel_dev = abs(observed_dr - predicted_pd) / predicted_pd

        if rel_dev <= green_threshold:
            zone = "green"
        elif rel_dev <= amber_threshold:
            zone = "amber"
        else:
            zone = "red"

        return {
            "predicted_pd": predicted_pd,
            "observed_dr": observed_dr,
            "n_obligors": n_obligors,
            "relative_deviation": round(rel_dev, 4),
            "zone": zone,
            "interpretation": self.TRAFFIC_LIGHT_ZONES[zone],
        }

    def binomial_test(
        self,
        predicted_pd: float,
        observed_dr: float,
        n_obligors: int,
        significance_level: float = 0.05,
    ) -> dict:
        """One-sided binomial test: is the observed DR significantly above predicted PD?

        Tests H0: true PD = predicted_pd vs H1: true PD > predicted_pd.
        A rejection indicates the model is underestimating risk.

        Parameters
        ----------
        predicted_pd : float — model TTC PD.
        observed_dr : float — realised default rate.
        n_obligors : int — portfolio size.
        significance_level : float — alpha for the test.

        Returns
        -------
        dict with keys: p_value, reject_h0, observed_defaults,
            expected_defaults, interpretation.
        """
        observed_defaults = int(round(observed_dr * n_obligors))
        p_value = 1 - stats.binom.cdf(observed_defaults - 1, n_obligors, predicted_pd)

        return {
            "predicted_pd": predicted_pd,
            "observed_dr": observed_dr,
            "n_obligors": n_obligors,
            "observed_defaults": observed_defaults,
            "expected_defaults": round(predicted_pd * n_obligors, 2),
            "p_value": round(p_value, 6),
            "significance_level": significance_level,
            "reject_h0": bool(p_value < significance_level),
            "interpretation": (
                "Model likely underestimates risk — recalibration advised."
                if p_value < significance_level
                else "No significant evidence of model underestimation."
            ),
        }

    def portfolio_backtest(
        self,
        backtest_df: pd.DataFrame,
        pd_col: str = "predicted_pd",
        dr_col: str = "observed_dr",
        n_col: str = "n_obligors",
    ) -> pd.DataFrame:
        """Run traffic light and binomial tests across all grades/segments.

        Parameters
        ----------
        backtest_df : pd.DataFrame — one row per grade or segment.
        pd_col, dr_col, n_col : str — column names.

        Returns
        -------
        pd.DataFrame — results with zone and p-value per row.
        """
        results = []
        for _, row in backtest_df.iterrows():
            tl = self.traffic_light_test(row[pd_col], row[dr_col], int(row[n_col]))
            bt = self.binomial_test(row[pd_col], row[dr_col], int(row[n_col]))
            results.append({
                "predicted_pd": row[pd_col],
                "observed_dr": row[dr_col],
                "n_obligors": int(row[n_col]),
                "relative_deviation": tl["relative_deviation"],
                "zone": tl["zone"],
                "p_value": bt["p_value"],
                "reject_h0": bt["reject_h0"],
            })
        return pd.DataFrame(results)
