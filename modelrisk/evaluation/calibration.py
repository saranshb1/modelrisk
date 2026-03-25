"""Calibration metrics: Hosmer-Lemeshow test and reliability analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


class CalibrationMetrics:
    """Model calibration assessment for probabilistic risk models.

    Calibration measures whether predicted probabilities match observed
    default rates. A model may discriminate well (high Gini) but still
    be poorly calibrated (systematically over/under-predicting PD).

    Includes:
    - Hosmer-Lemeshow (HL) goodness-of-fit test
    - Reliability diagram (calibration curve) data
    - Mean predicted vs observed rate by decile
    - Expected vs Actual (EVA) analysis

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Binary outcomes (1 = default, 0 = non-default).
    y_score : array-like of shape (n_samples,)
        Predicted probabilities.
    n_bins : int
        Number of bins for decile/percentile analysis.

    Examples
    --------
    >>> cal = CalibrationMetrics(y_true, y_pred_proba, n_bins=10)
    >>> cal.hosmer_lemeshow()
    >>> cal.reliability_diagram_data()
    >>> cal.summary()
    """

    def __init__(
        self,
        y_true: pd.Series | np.ndarray,
        y_score: pd.Series | np.ndarray,
        n_bins: int = 10,
    ) -> None:
        self.y_true = np.asarray(y_true, dtype=float)
        self.y_score = np.asarray(y_score, dtype=float)
        self.n_bins = n_bins

        if len(self.y_true) != len(self.y_score):
            raise ValueError("y_true and y_score must have the same length.")

    # ------------------------------------------------------------------
    # Hosmer-Lemeshow test
    # ------------------------------------------------------------------

    def hosmer_lemeshow(self) -> dict:
        """Hosmer-Lemeshow goodness-of-fit test.

        Groups observations into n_bins quantile bins by predicted probability.
        Computes a chi-squared statistic comparing observed vs expected defaults
        within each bin.

        Returns
        -------
        dict with keys:
            statistic : float
                HL chi-squared statistic.
            p_value : float
                p-value (high p-value = good calibration; conventionally
                p > 0.05 means no evidence of poor fit).
            df : int
                Degrees of freedom (n_bins - 2).
            bins : pd.DataFrame
                Per-bin expected vs observed counts.
            interpretation : str
        """
        df = pd.DataFrame({"prob": self.y_score, "obs": self.y_true})
        df["bin"] = pd.qcut(df["prob"], q=self.n_bins, duplicates="drop", labels=False)

        hl_stat = 0.0
        bins_data = []
        for bin_id, group in df.groupby("bin"):
            n = len(group)
            observed_events = group["obs"].sum()
            expected_events = group["prob"].sum()
            observed_non = n - observed_events
            expected_non = n - expected_events

            expected_events = max(expected_events, 1e-6)
            expected_non = max(expected_non, 1e-6)

            hl_stat += (observed_events - expected_events) ** 2 / expected_events
            hl_stat += (observed_non - expected_non) ** 2 / expected_non

            bins_data.append(
                {
                    "bin": bin_id,
                    "n": n,
                    "observed_defaults": int(observed_events),
                    "expected_defaults": round(expected_events, 2),
                    "observed_rate": round(observed_events / n, 4),
                    "expected_rate": round(expected_events / n, 4),
                }
            )

        dof = max(self.n_bins - 2, 1)
        p_value = float(1 - stats.chi2.cdf(hl_stat, df=dof))
        interpretation = (
            "Good calibration (p > 0.05)" if p_value > 0.05
            else "Poor calibration detected (p ≤ 0.05)"
        )

        return {
            "statistic": float(hl_stat),
            "p_value": p_value,
            "df": dof,
            "bins": pd.DataFrame(bins_data),
            "interpretation": interpretation,
        }

    # ------------------------------------------------------------------
    # Reliability diagram
    # ------------------------------------------------------------------

    def reliability_diagram_data(self) -> pd.DataFrame:
        """Compute data for a reliability (calibration) diagram.

        Each bin's mean predicted probability is plotted against its
        observed event rate. Perfect calibration → points on the diagonal.

        Returns
        -------
        pd.DataFrame with columns: bin_mean_predicted, observed_rate, count.
        """
        df = pd.DataFrame({"prob": self.y_score, "obs": self.y_true})
        df["bin"] = pd.cut(df["prob"], bins=self.n_bins)

        rows = []
        for bin_id, group in df.groupby("bin", observed=True):
            if len(group) == 0:
                continue
            rows.append(
                {
                    "bin_mean_predicted": group["prob"].mean(),
                    "observed_rate": group["obs"].mean(),
                    "count": len(group),
                }
            )
        return pd.DataFrame(rows).sort_values("bin_mean_predicted").reset_index(drop=True)

    # ------------------------------------------------------------------
    # Expected vs Actual (EVA)
    # ------------------------------------------------------------------

    def expected_vs_actual(self) -> pd.DataFrame:
        """Decile-level Expected vs Actual default rates.

        Useful for regulatory model validation reports.

        Returns
        -------
        pd.DataFrame with per-decile expected and observed default rates.
        """
        df = pd.DataFrame({"prob": self.y_score, "obs": self.y_true})
        df["decile"] = pd.qcut(df["prob"], q=10, duplicates="drop", labels=False)

        rows = []
        for decile, group in df.groupby("decile"):
            rows.append(
                {
                    "decile": int(decile) + 1,
                    "n": len(group),
                    "mean_predicted_pd": round(group["prob"].mean(), 4),
                    "observed_default_rate": round(group["obs"].mean(), 4),
                    "ratio_actual_to_predicted": round(
                        group["obs"].mean() / max(group["prob"].mean(), 1e-6), 3
                    ),
                }
            )
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Scalar calibration metrics
    # ------------------------------------------------------------------

    def mean_calibration_error(self) -> float:
        """Mean Calibration Error — average absolute bin deviation."""
        rd = self.reliability_diagram_data()
        return float(np.mean(np.abs(rd["bin_mean_predicted"] - rd["observed_rate"])))

    def expected_calibration_error(self) -> float:
        """Expected Calibration Error (ECE) — weighted by bin size."""
        rd = self.reliability_diagram_data()
        n_total = rd["count"].sum()
        ece = np.sum(
            rd["count"] / n_total * np.abs(rd["bin_mean_predicted"] - rd["observed_rate"])
        )
        return float(ece)

    def overall_default_rate_ratio(self) -> float:
        """Ratio of mean predicted PD to observed default rate.

        A value of 1.0 indicates perfect calibration at the portfolio level.
        > 1 = model over-predicts risk; < 1 = model under-predicts risk.
        """
        obs_rate = self.y_true.mean()
        pred_rate = self.y_score.mean()
        return float(pred_rate / max(obs_rate, 1e-9))

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> pd.DataFrame:
        """Return all calibration metrics as a tidy DataFrame."""
        hl = self.hosmer_lemeshow()
        rows = [
            ("HL statistic", hl["statistic"], f"Chi-squared, df={hl['df']}"),
            ("HL p-value", hl["p_value"], hl["interpretation"]),
            (
                "Mean calibration error",
                self.mean_calibration_error(),
                "Avg |predicted - observed| per bin",
            ),
            (
                "Expected calibration error",
                self.expected_calibration_error(),
                "Weighted avg |predicted - observed|",
            ),
            (
                "Portfolio rate ratio",
                self.overall_default_rate_ratio(),
                "Mean predicted PD / observed rate; 1.0 = perfect",
            ),
        ]
        return pd.DataFrame(rows, columns=["metric", "value", "interpretation"])
