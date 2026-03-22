"""Scorecard construction with Weight of Evidence (WoE) and Information Value (IV)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


class Scorecard:
    """Credit scorecard builder using Weight of Evidence encoding.

    Transforms categorical and binned continuous features into WoE-encoded
    values, computes Information Value for feature selection, then fits a
    logistic regression to produce a points-based scorecard.

    Parameters
    ----------
    pdo : int
        Points to double the odds (standard: 20).
    base_score : int
        Score at the base odds (standard: 600).
    base_odds : float
        Odds at the base score (standard: 1/19 ≈ 50:1 goods to bads).

    Examples
    --------
    >>> sc = Scorecard(pdo=20, base_score=600, base_odds=1/19)
    >>> sc.fit(X_binned, y)
    >>> scores = sc.score(X_binned)
    >>> sc.information_value_summary()
    """

    def __init__(
        self,
        pdo: int = 20,
        base_score: int = 600,
        base_odds: float = 1 / 19,
    ) -> None:
        self.pdo = pdo
        self.base_score = base_score
        self.base_odds = base_odds
        self._factor = pdo / np.log(2)
        self._offset = base_score - self._factor * np.log(base_odds)
        self.woe_tables_: dict[str, pd.DataFrame] = {}
        self.iv_: dict[str, float] = {}
        self._model: LogisticRegression | None = None
        self.feature_names_: list[str] = []

    # ------------------------------------------------------------------
    # WoE / IV calculation
    # ------------------------------------------------------------------

    def _compute_woe_table(self, series: pd.Series, y: pd.Series) -> pd.DataFrame:
        """Compute WoE and IV for a single categorical/binned feature."""
        df = pd.DataFrame({"bin": series, "target": y})
        total_bads = (y == 1).sum()
        total_goods = (y == 0).sum()

        rows = []
        for bin_val, group in df.groupby("bin", observed=True):
            bads = (group["target"] == 1).sum()
            goods = (group["target"] == 0).sum()
            dist_bad = bads / total_bads if total_bads > 0 else 1e-6
            dist_good = goods / total_goods if total_goods > 0 else 1e-6
            dist_bad = max(dist_bad, 1e-6)
            dist_good = max(dist_good, 1e-6)
            woe = np.log(dist_good / dist_bad)
            iv = (dist_good - dist_bad) * woe
            rows.append(
                {
                    "bin": bin_val,
                    "count": len(group),
                    "bads": bads,
                    "goods": goods,
                    "bad_rate": bads / len(group) if len(group) > 0 else 0,
                    "dist_bad": dist_bad,
                    "dist_good": dist_good,
                    "woe": woe,
                    "iv": iv,
                }
            )
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Fit / transform
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> "Scorecard":
        """Fit WoE tables and the underlying logistic regression.

        Parameters
        ----------
        X : pd.DataFrame
            Pre-binned feature matrix (each column should be categorical or
            ordinal bins — use pd.cut / pd.qcut before calling fit).
        y : array-like of shape (n_samples,)
            Binary target (1 = bad/default, 0 = good/non-default).

        Returns
        -------
        self
        """
        y_s = pd.Series(np.asarray(y), name="target")
        self.feature_names_ = list(X.columns)

        for col in self.feature_names_:
            tbl = self._compute_woe_table(X[col], y_s)
            self.woe_tables_[col] = tbl
            self.iv_[col] = float(tbl["iv"].sum())

        X_woe = self._apply_woe(X)
        self._model = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
        self._model.fit(X_woe.values, y_s.values)
        return self

    def _apply_woe(self, X: pd.DataFrame) -> pd.DataFrame:
        """Encode features using fitted WoE tables."""
        result = pd.DataFrame(index=X.index)
        for col in self.feature_names_:
            tbl = self.woe_tables_[col].set_index("bin")["woe"]
            mapped = X[col].astype(object).map(tbl)
            result[col] = pd.to_numeric(mapped, errors="coerce").fillna(0.0)
        return result

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return predicted PD from the scorecard model."""
        if self._model is None:
            raise RuntimeError("Scorecard has not been fitted yet.")
        X_woe = self._apply_woe(X)
        return self._model.predict_proba(X_woe.values)[:, 1]

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """Convert predicted probabilities to scorecard points.

        Higher scores indicate lower risk (better creditworthiness).

        Parameters
        ----------
        X : pd.DataFrame
            Pre-binned feature matrix.

        Returns
        -------
        np.ndarray of integer scorecard points.
        """
        proba = self.predict_proba(X)
        odds = (1 - proba) / np.clip(proba, 1e-9, None)
        scores = self._offset + self._factor * np.log(odds)
        return np.round(scores).astype(int)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def information_value_summary(self) -> pd.DataFrame:
        """Return IV summary for all features with predictive power labels.

        Returns
        -------
        pd.DataFrame sorted by IV descending.
        """
        def _label(iv: float) -> str:
            if iv < 0.02:
                return "Useless"
            elif iv < 0.1:
                return "Weak"
            elif iv < 0.3:
                return "Medium"
            elif iv < 0.5:
                return "Strong"
            return "Very strong / suspicious"

        rows = [
            {"feature": feat, "iv": iv, "predictive_power": _label(iv)}
            for feat, iv in self.iv_.items()
        ]
        return pd.DataFrame(rows).sort_values("iv", ascending=False).reset_index(drop=True)

    def woe_summary(self, feature: str) -> pd.DataFrame:
        """Return the WoE table for a specific feature.

        Parameters
        ----------
        feature : str
            Column name.
        """
        if feature not in self.woe_tables_:
            raise KeyError(f"Feature '{feature}' not found. Fitted features: {self.feature_names_}")
        return self.woe_tables_[feature].copy()
