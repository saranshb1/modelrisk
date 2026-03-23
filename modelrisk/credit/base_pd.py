"""Abstract base class for all PD models in modelrisk.credit.

Both IFRS 9 (PIT) and IRB (TTC) pipelines accept any model that
satisfies this interface, allowing statistical model choice to remain
independent of the regulatory calibration layer applied on top.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BasePDModel(ABC):
    """Abstract base for all probability-of-default models.

    Concrete subclasses: ``LogisticPD``, ``RandomForestPD``, ``XGBoostPD``.

    Enforces three methods required by both IFRS 9 and IRB calibration
    pipelines so that any model can be dropped in without touching the
    pipeline code.
    """

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
    ) -> "BasePDModel":
        """Fit the model on training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,) — binary default indicator.

        Returns
        -------
        self
        """

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Return raw predicted default probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        np.ndarray of shape (n_samples,) in [0, 1].
        """

    @abstractmethod
    def feature_importance_summary(self) -> pd.DataFrame:
        """Return feature importances as a tidy DataFrame.

        Returns
        -------
        pd.DataFrame with at minimum columns: feature, importance, importance_pct.
        """
