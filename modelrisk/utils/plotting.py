"""Risk-domain plotting utilities built on matplotlib."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score

try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False


def _require_mpl() -> None:
    if not _MPL_AVAILABLE:
        raise ImportError("matplotlib is required for plotting. pip install matplotlib")


class RiskPlotter:
    """Collection of risk model visualisation helpers.

    All methods return a ``matplotlib.figure.Figure`` so callers can
    save or display them as needed.

    Parameters
    ----------
    figsize : tuple
        Default figure size (width, height) in inches.
    style : str
        Matplotlib style (e.g. 'seaborn-v0_8-whitegrid').

    Examples
    --------
    >>> plotter = RiskPlotter()
    >>> fig = plotter.roc_curve(y_true, y_score)
    >>> fig.savefig("roc.png", dpi=150)
    """

    def __init__(self, figsize: tuple = (8, 6), style: str = "seaborn-v0_8-whitegrid") -> None:
        _require_mpl()
        self.figsize = figsize
        try:
            plt.style.use(style)
        except OSError:
            pass  # fall back to default

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def roc_curve(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        label: str = "Model",
    ):
        """Plot ROC curve with AUC and Gini annotations."""

        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)
        gini = 2 * auc - 1

        fig, ax = plt.subplots(figsize=self.figsize)
        ax.plot(fpr, tpr, lw=2, label=f"{label} (AUC={auc:.3f}, Gini={gini:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend(loc="lower right")
        fig.tight_layout()
        return fig

    def cap_curve(self, y_true: np.ndarray, y_score: np.ndarray, label: str = "Model"):
        """Plot CAP (Cumulative Accuracy Profile) curve."""
        df = pd.DataFrame({"score": y_score, "target": y_true})
        df_sorted = df.sort_values("score", ascending=False).reset_index(drop=True)
        n = len(df_sorted)
        total_events = df_sorted["target"].sum()

        pct_pop = np.arange(1, n + 1) / n
        pct_events = df_sorted["target"].cumsum() / total_events
        perfect_pct = np.minimum(pct_pop / (total_events / n), 1.0)

        fig, ax = plt.subplots(figsize=self.figsize)
        ax.plot(pct_pop, pct_events, lw=2, label=label)
        ax.plot(pct_pop, perfect_pct, "g--", lw=1.5, label="Perfect model")
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
        ax.set_xlabel("% of Population")
        ax.set_ylabel("% of Defaults Captured")
        ax.set_title("Cumulative Accuracy Profile (CAP)")
        ax.legend()
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        fig.tight_layout()
        return fig

    def ks_plot(self, y_true: np.ndarray, y_score: np.ndarray):
        """Plot KS separation between default and non-default score distributions."""
        scores_bad = np.sort(y_score[y_true == 1])
        scores_good = np.sort(y_score[y_true == 0])

        fig, ax = plt.subplots(figsize=self.figsize)
        x = np.linspace(0, 1, 200)
        cdf_bad = np.searchsorted(scores_bad, x, side="right") / len(scores_bad)
        cdf_good = np.searchsorted(scores_good, x, side="right") / len(scores_good)
        ks = float(np.max(np.abs(cdf_bad - cdf_good)))

        ax.plot(x, cdf_bad, label="Defaults (bad)", lw=2)
        ax.plot(x, cdf_good, label="Non-defaults (good)", lw=2)
        ks_x = x[np.argmax(np.abs(cdf_bad - cdf_good))]
        ax.axvline(ks_x, color="red", linestyle="--", lw=1.5, label=f"KS = {ks:.3f}")
        ax.set_xlabel("Score")
        ax.set_ylabel("Cumulative Proportion")
        ax.set_title("KS Separation Plot")
        ax.legend()
        fig.tight_layout()
        return fig

    def reliability_diagram(self, y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10):
        """Plot reliability (calibration) diagram."""
        from modelrisk.evaluation.calibration import CalibrationMetrics
        cal = CalibrationMetrics(y_true, y_score, n_bins=n_bins)
        rd = cal.reliability_diagram_data()

        fig, ax = plt.subplots(figsize=self.figsize)
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
        ax.scatter(
            rd["bin_mean_predicted"], rd["observed_rate"],
            s=rd["count"] / rd["count"].max() * 200,
            alpha=0.7, label="Model (size ∝ bin count)"
        )
        ax.plot(rd["bin_mean_predicted"], rd["observed_rate"], alpha=0.5)
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Observed Event Rate")
        ax.set_title("Reliability Diagram (Calibration Curve)")
        ax.legend()
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Market / Loss distributions
    # ------------------------------------------------------------------

    def loss_distribution(
        self,
        losses: np.ndarray,
        var_level: float | None = None,
        cvar_level: float | None = None,
        title: str = "Loss Distribution",
    ):
        """Plot histogram of simulated or historical losses with VaR/CVaR lines."""
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.hist(losses, bins=80,
                density=True, alpha=0.7,
                color="steelblue", label="Loss distribution"
                )

        if var_level is not None:
            var = float(np.quantile(losses, var_level))
            ax.axvline(var, color="orange",
                        lw=2, linestyle="--",
                        label=f"VaR ({var_level:.1%}): {var:,.0f}")

        if cvar_level is not None:
            var_c = float(np.quantile(losses, cvar_level))
            cvar = float(np.mean(losses[losses >= var_c]))
            #Add changes to fix CI E501 line too long error
            ax.axvline(cvar, 
                        color="red",
                        lw=2, linestyle="--",
                        label=f"CVaR ({cvar_level:.1%}): {cvar:,.0f}"
                        )

        ax.set_xlabel("Loss")
        ax.set_ylabel("Density")
        ax.set_title(title)
        ax.legend()
        fig.tight_layout()
        return fig

    def volatility_series(self, vol: pd.Series, title: str = "Conditional Volatility"):
        """Plot a volatility time series (e.g. from GARCH or EWMA)."""
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.plot(vol.values, lw=1.2, color="steelblue")
        ax.set_xlabel("Time")
        ax.set_ylabel("Annualised Volatility")
        ax.set_title(title)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Explainability
    # ------------------------------------------------------------------

    def shap_summary(self, shap_values: pd.DataFrame, max_features: int = 15):
        """Horizontal bar plot of mean absolute SHAP values."""
        mean_shap = shap_values.abs().mean().sort_values(ascending=True).tail(max_features)
        fig, ax = plt.subplots(figsize=(self.figsize[0], max(5, len(mean_shap) * 0.4)))
        ax.barh(mean_shap.index, mean_shap.values, color="steelblue")
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title("Feature Importance (Mean |SHAP|)")
        fig.tight_layout()
        return fig

    def waterfall(self, shap_values: pd.Series, base_value: float, prediction: float):
        """SHAP waterfall plot for a single instance."""
        sv = shap_values.sort_values(key=abs, ascending=False).head(15)
        features = sv.index.tolist()
        values = sv.values

        cumulative = np.concatenate([[base_value], base_value + np.cumsum(values)])
        colours = ["#ef4444" if v >= 0 else "#22c55e" for v in values]

        fig, ax = plt.subplots(figsize=(self.figsize[0], max(5, len(features) * 0.5)))
        for i, (feat, val, col) in enumerate(zip(features, values, colours)):
            ax.barh(
                i, abs(val),
                left=min(cumulative[i], cumulative[i + 1]),
                color=col, alpha=0.85
            )
            ax.text(cumulative[i + 1], i, f" {val:+.4f}", va="center", fontsize=9)

        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features)
        ax.axvline(base_value, color="black", lw=1, linestyle="--", label=f"Base: {base_value:.4f}")
        ax.axvline(
                    prediction,
                    color="blue", lw=1.5, linestyle="-",
                    label=f"Prediction: {prediction:.4f}"
                    )
        ax.set_xlabel("Prediction")
        ax.set_title("SHAP Waterfall")
        ax.legend()
        fig.tight_layout()
        return fig
