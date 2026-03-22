"""Classification model evaluation metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn import metrics as skmetrics


class ClassificationMetrics:
    """Comprehensive classification metrics for binary risk models.

    Computes a full suite of discrimination and ranking metrics commonly
    used in credit and market risk model validation:
    - AUC-ROC and Gini coefficient
    - KS (Kolmogorov-Smirnov) statistic
    - F1 score, Precision, Recall
    - Accuracy, Balanced accuracy
    - Matthews Correlation Coefficient (MCC)
    - Brier Score
    - Lift and CAP curve metrics

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Binary ground truth (1 = default/event, 0 = non-event).
    y_score : array-like of shape (n_samples,)
        Predicted probabilities of the positive class.
    threshold : float
        Decision threshold for converting probabilities to class labels.

    Examples
    --------
    >>> cm = ClassificationMetrics(y_true, y_pred_proba)
    >>> cm.summary()
    >>> cm.gini()
    >>> cm.ks_statistic()
    """

    def __init__(
        self,
        y_true: pd.Series | np.ndarray,
        y_score: pd.Series | np.ndarray,
        threshold: float = 0.5,
    ) -> None:
        self.y_true = np.asarray(y_true, dtype=int)
        self.y_score = np.asarray(y_score, dtype=float)
        self.threshold = threshold
        self.y_pred = (self.y_score >= threshold).astype(int)

        if len(np.unique(self.y_true)) < 2:
            raise ValueError("y_true must contain both classes (0 and 1).")

    # ------------------------------------------------------------------
    # Core metrics
    # ------------------------------------------------------------------

    def auc_roc(self) -> float:
        """Area Under the ROC Curve."""
        return float(skmetrics.roc_auc_score(self.y_true, self.y_score))

    def gini(self) -> float:
        """Gini coefficient = 2 * AUC - 1.

        Ranges from -1 (perfectly wrong) to 1 (perfectly discriminating).
        A well-calibrated credit model typically achieves Gini > 0.4.
        """
        return 2.0 * self.auc_roc() - 1.0

    def ks_statistic(self) -> dict:
        """Kolmogorov-Smirnov statistic.

        Measures the maximum separation between the cumulative default and
        non-default score distributions. KS > 0.3 is generally considered
        acceptable for retail credit models.

        Returns
        -------
        dict with keys: ks, threshold_at_ks
        """
        fpr, tpr, thresholds = skmetrics.roc_curve(self.y_true, self.y_score)
        ks_values = tpr - fpr
        idx = np.argmax(ks_values)
        return {
            "ks": float(ks_values[idx]),
            "threshold_at_ks": float(thresholds[idx]),
        }

    def f1_score(self, average: str = "binary") -> float:
        """F1 score — harmonic mean of precision and recall."""
        return float(skmetrics.f1_score(self.y_true, self.y_pred, average=average))

    def precision(self) -> float:
        """Precision = TP / (TP + FP)."""
        return float(skmetrics.precision_score(self.y_true, self.y_pred, zero_division=0))

    def recall(self) -> float:
        """Recall (Sensitivity) = TP / (TP + FN)."""
        return float(skmetrics.recall_score(self.y_true, self.y_pred, zero_division=0))

    def specificity(self) -> float:
        """Specificity (True Negative Rate) = TN / (TN + FP)."""
        tn, fp, fn, tp = skmetrics.confusion_matrix(self.y_true, self.y_pred).ravel()
        return float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    def accuracy(self) -> float:
        """Overall classification accuracy."""
        return float(skmetrics.accuracy_score(self.y_true, self.y_pred))

    def balanced_accuracy(self) -> float:
        """Balanced accuracy — average of recall for each class.

        Preferred when class imbalance is significant (typical in credit).
        """
        return float(skmetrics.balanced_accuracy_score(self.y_true, self.y_pred))

    def mcc(self) -> float:
        """Matthews Correlation Coefficient.

        Ranges from -1 to +1; robust to class imbalance.
        """
        return float(skmetrics.matthews_corrcoef(self.y_true, self.y_pred))

    def brier_score(self) -> float:
        """Brier Score — mean squared error of probability predictions.

        Lower is better. A model predicting the base rate everywhere achieves
        Brier = base_rate * (1 - base_rate).
        """
        return float(skmetrics.brier_score_loss(self.y_true, self.y_score))

    def log_loss(self) -> float:
        """Logarithmic loss (cross-entropy)."""
        return float(skmetrics.log_loss(self.y_true, self.y_score))

    # ------------------------------------------------------------------
    # CAP / Lift
    # ------------------------------------------------------------------

    def cap_accuracy_ratio(self) -> float:
        """Accuracy Ratio from the CAP (Cumulative Accuracy Profile) curve.

        Equivalent to the Gini coefficient when using predicted probabilities.
        """
        return self.gini()

    def lift_at_decile(self, decile: int = 1) -> float:
        """Lift at a given decile of predicted scores.

        Parameters
        ----------
        decile : int
            Decile (1 = top 10% of predicted scores by risk).

        Returns
        -------
        float : lift = (% of events captured in decile) / (% of population in decile)
        """
        if not 1 <= decile <= 10:
            raise ValueError("decile must be between 1 and 10.")
        df = pd.DataFrame({"score": self.y_score, "target": self.y_true})
        df = df.sort_values("score", ascending=False).reset_index(drop=True)
        cutoff = int(np.ceil(len(df) * decile / 10))
        top_events = df.iloc[:cutoff]["target"].sum()
        total_events = df["target"].sum()
        if total_events == 0:
            return 0.0
        return float((top_events / total_events) / (decile / 10))

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> pd.DataFrame:
        """Return all metrics as a tidy DataFrame.

        Returns
        -------
        pd.DataFrame with columns: metric, value, interpretation.
        """
        ks = self.ks_statistic()
        rows = [
            ("AUC-ROC", self.auc_roc(), "Discrimination; >0.7 acceptable, >0.8 good"),
            ("Gini", self.gini(), "2*AUC - 1; >0.4 acceptable for credit"),
            ("KS statistic", ks["ks"], ">0.3 acceptable; measures score separation"),
            ("KS threshold", ks["threshold_at_ks"], "Score at maximum separation"),
            ("F1 score", self.f1_score(), "Harmonic mean of precision and recall"),
            ("Precision", self.precision(), "TP / (TP + FP)"),
            ("Recall / Sensitivity", self.recall(), "TP / (TP + FN)"),
            ("Specificity", self.specificity(), "TN / (TN + FP)"),
            ("Accuracy", self.accuracy(), "Overall correct classifications"),
            ("Balanced accuracy", self.balanced_accuracy(), "Average recall per class"),
            ("MCC", self.mcc(), "Robust to imbalance; range [-1, 1]"),
            ("Brier score", self.brier_score(), "Probability calibration; lower is better"),
            ("Log loss", self.log_loss(), "Cross-entropy; lower is better"),
            ("Lift @ decile 1", self.lift_at_decile(1), "Capture rate in top 10% of risk scores"),
        ]
        return pd.DataFrame(rows, columns=["metric", "value", "interpretation"])

    # ------------------------------------------------------------------
    # Curve data (for plotting)
    # ------------------------------------------------------------------

    def roc_curve_data(self) -> pd.DataFrame:
        """ROC curve data for plotting."""
        fpr, tpr, thresholds = skmetrics.roc_curve(self.y_true, self.y_score)
        return pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds})

    def pr_curve_data(self) -> pd.DataFrame:
        """Precision-Recall curve data for plotting."""
        precision, recall, thresholds = skmetrics.precision_recall_curve(
            self.y_true, self.y_score
        )
        return pd.DataFrame({
            "precision": precision,
            "recall": recall,
            "threshold": np.append(thresholds, np.nan),
        })

    def cap_curve_data(self) -> pd.DataFrame:
        """CAP (Cumulative Accuracy Profile) curve data."""
        df = pd.DataFrame({"score": self.y_score, "target": self.y_true})
        df = df.sort_values("score", ascending=False).reset_index(drop=True)
        n = len(df)
        total_events = df["target"].sum()
        pct_population = np.arange(1, n + 1) / n
        pct_events = df["target"].cumsum() / total_events
        return pd.DataFrame({"pct_population": pct_population, "pct_events": pct_events})
