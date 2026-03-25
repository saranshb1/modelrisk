"""Probability of Default (PD) models.

Available models
----------------
LogisticPD      Logistic regression (interpretable baseline)
RandomForestPD  Random forest ensemble (non-linear, robust to outliers)
XGBoostPD       Gradient boosted trees (typically highest accuracy)
MertonPD        Structural model based on Merton (1974)

All statistical models share a common interface::

    model.fit(X, y)                   -> self
    model.predict_proba(X)            -> np.ndarray
    model.feature_importance_summary() -> pd.DataFrame

LogisticPD additionally exposes ``coefficient_summary()``.
RandomForestPD exposes ``tree_depth_summary()``, ``permutation_importance()``,
``partial_dependence()``, and ``oob_score_``.
XGBoostPD exposes ``learning_curve_data()``, ``best_iteration()``,
``parameter_summary()``, and a richer ``feature_importance_summary(importance_type=...)``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Shared mixin
# ---------------------------------------------------------------------------

class _TreeMixin:
    """Mixin for tree-based models — default feature_importance_summary."""

    feature_names_: list[str] | None

    def feature_importance_summary(self) -> pd.DataFrame:
        if not hasattr(self._model, "feature_importances_"):
            raise RuntimeError("Model has not been fitted yet.")
        importances = self._model.feature_importances_
        names = self.feature_names_ or [f"x{i}" for i in range(len(importances))]
        total = importances.sum()
        return (
            pd.DataFrame({
                "feature": names,
                "importance": importances,
                "importance_pct": importances / total * 100 if total > 0 else importances,
            })
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    @staticmethod
    def _to_array(X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return X.values if isinstance(X, pd.DataFrame) else np.asarray(X)


# ---------------------------------------------------------------------------
# LogisticPD
# ---------------------------------------------------------------------------

class LogisticPD:
    """Logistic regression PD model.

    Parameters
    ----------
    C : float
        Inverse regularisation strength.
    max_iter : int
    scale_features : bool
        Standardise features before fitting (recommended).

    Examples
    --------
    >>> model = LogisticPD()
    >>> model.fit(X_train, y_train)
    >>> model.predict_proba(X_test)
    >>> model.coefficient_summary()
    >>> model.feature_importance_summary()
    """

    def __init__(self, C: float = 1.0, max_iter: int = 1000, scale_features: bool = True) -> None:
        self.C = C
        self.max_iter = max_iter
        self.scale_features = scale_features
        self._model = LogisticRegression(C=C, max_iter=max_iter, solver="lbfgs")
        self._scaler = StandardScaler() if scale_features else None
        self.feature_names_: list[str] | None = None
        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> "LogisticPD":
        """Fit the logistic PD model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,) — binary default indicator.

        Returns
        -------
        self
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            X_arr = X.values
        else:
            X_arr = np.asarray(X)
        y_arr = np.asarray(y)
        if self._scaler is not None:
            X_arr = self._scaler.fit_transform(X_arr)
        self._model.fit(X_arr, y_arr)
        self.coef_ = self._model.coef_[0]
        self.intercept_ = float(self._model.intercept_[0])
        return self

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Return predicted default probabilities (shape n_samples,)."""
        X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        if self._scaler is not None:
            X_arr = self._scaler.transform(X_arr)
        return self._model.predict_proba(X_arr)[:, 1]

    def coefficient_summary(self) -> pd.DataFrame:
        """Feature coefficients and odds ratios, sorted by |coefficient|.

        Returns
        -------
        pd.DataFrame — columns: feature, coefficient, odds_ratio.

        Raises
        ------
        RuntimeError if model not fitted.
        """
        if self.coef_ is None:
            raise RuntimeError("Model has not been fitted yet.")
        names = self.feature_names_ or [f"x{i}" for i in range(len(self.coef_))]
        return (
            pd.DataFrame({
                "feature": names,
                "coefficient": self.coef_,
                "odds_ratio": np.exp(self.coef_),
            })
            .sort_values("coefficient", key=abs, ascending=False)
            .reset_index(drop=True)
        )

    def feature_importance_summary(self) -> pd.DataFrame:
        """Feature importance via absolute coefficient magnitude.

        Provides a common interface consistent with RandomForestPD and
        XGBoostPD so all three models can be compared side-by-side.

        Returns
        -------
        pd.DataFrame — columns: feature, importance (|coef|), importance_pct.
        """
        if self.coef_ is None:
            raise RuntimeError("Model has not been fitted yet.")
        names = self.feature_names_ or [f"x{i}" for i in range(len(self.coef_))]
        abs_coef = np.abs(self.coef_)
        total = abs_coef.sum()
        return (
            pd.DataFrame({
                "feature": names,
                "importance": abs_coef,
                "importance_pct": abs_coef / total * 100 if total > 0 else abs_coef,
            })
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )


# ---------------------------------------------------------------------------
# RandomForestPD
# ---------------------------------------------------------------------------

class RandomForestPD(_TreeMixin):
    """Random forest ensemble PD model.

    Handles class imbalance via ``class_weight='balanced'`` by default —
    standard practice on credit datasets where defaults are rare events.

    Shares the ``fit`` / ``predict_proba`` / ``feature_importance_summary``
    interface with ``LogisticPD`` and ``XGBoostPD``.

    Parameters
    ----------
    n_estimators : int
        Number of trees in the forest.
    max_depth : int or None
        Maximum tree depth. ``None`` grows until leaves are pure.
    min_samples_leaf : int
        Minimum samples required at each leaf (acts as regularisation).
    max_features : str or float
        Features to consider per split. ``'sqrt'`` is the sklearn default.
    class_weight : str, dict, or None
        ``'balanced'`` corrects for imbalanced default rates.
    oob_score : bool
        Whether to compute out-of-bag classification accuracy.
    n_jobs : int — ``-1`` uses all cores.
    random_state : int or None

    Examples
    --------
    >>> model = RandomForestPD(n_estimators=300, max_depth=8)
    >>> model.fit(X_train, y_train)
    >>> model.predict_proba(X_test)
    >>> model.feature_importance_summary()
    >>> model.tree_depth_summary()
    >>> model.oob_score_
    >>> model.permutation_importance(X_val, y_val)
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int | None = None,
        min_samples_leaf: int = 20,
        max_features: str | float = "sqrt",
        class_weight: str | dict | None = "balanced",
        oob_score: bool = True,
        n_jobs: int = -1,
        random_state: int | None = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.class_weight = class_weight
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        self._model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            class_weight=class_weight,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
        )
        self.feature_names_: list[str] | None = None

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> "RandomForestPD":
        """Fit the random forest PD model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,) — binary default indicator.

        Returns
        -------
        self
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
        self._model.fit(self._to_array(X), np.asarray(y))
        return self

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Return predicted default probabilities (shape n_samples,)."""
        if not hasattr(self._model, "classes_"):
            raise RuntimeError("Model has not been fitted yet.")
        return self._model.predict_proba(self._to_array(X))[:, 1]

    # --- RandomForest-specific -------------------------------------------------

    @property
    def oob_score_(self) -> float | None:
        """Out-of-bag accuracy (proxy for generalisation; available if oob_score=True)."""
        if not self.oob_score:
            return None
        if not hasattr(self._model, "oob_score_"):
            raise RuntimeError("Model has not been fitted yet.")
        return float(self._model.oob_score_)

    def tree_depth_summary(self) -> pd.DataFrame:
        """Distribution of tree depths across all estimators.

        Very deep trees on credit data often signal overfitting.

        Returns
        -------
        pd.DataFrame — columns: statistic, value.

        Raises
        ------
        RuntimeError if model not fitted.
        """
        if not hasattr(self._model, "estimators_"):
            raise RuntimeError("Model has not been fitted yet.")
        depths = [est.get_depth() for est in self._model.estimators_]
        return pd.DataFrame({
            "statistic": ["min", "mean", "median", "max", "std"],
            "value": [
                int(np.min(depths)),
                round(float(np.mean(depths)), 2),
                float(np.median(depths)),
                int(np.max(depths)),
                round(float(np.std(depths)), 2),
            ],
        })

    def permutation_importance(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        n_repeats: int = 10,
        random_state: int | None = None,
    ) -> pd.DataFrame:
        """Permutation feature importance on a held-out validation set.

        More reliable than Gini importance (which is biased towards
        high-cardinality features). Measures actual AUC drop.

        Parameters
        ----------
        X : array-like — held-out features.
        y : array-like — held-out labels.
        n_repeats : int — shuffles per feature.
        random_state : int or None

        Returns
        -------
        pd.DataFrame — columns: feature, mean_importance, std_importance.
        """
        from sklearn.inspection import permutation_importance as _perm

        if not hasattr(self._model, "estimators_"):
            raise RuntimeError("Model has not been fitted yet.")
        result = _perm(
            self._model, self._to_array(X), np.asarray(y),
            n_repeats=n_repeats,
            random_state=random_state or self.random_state,
            scoring="roc_auc",
        )
        names = self.feature_names_ or [f"x{i}" for i in range(len(result.importances_mean))]
        return (
            pd.DataFrame({
                "feature": names,
                "mean_importance": result.importances_mean,
                "std_importance": result.importances_std,
            })
            .sort_values("mean_importance", ascending=False)
            .reset_index(drop=True)
        )

    def partial_dependence(
        self,
        X: pd.DataFrame | np.ndarray,
        feature: str | int,
        grid_resolution: int = 50,
    ) -> pd.DataFrame:
        """Marginal effect of a single feature on predicted PD.

        Averages out all other features to isolate the feature's
        individual contribution to predicted default probability.

        Parameters
        ----------
        X : array-like — representative sample (e.g. training data).
        feature : str or int — feature name (if feature_names_ set) or index.
        grid_resolution : int — number of grid points to evaluate.

        Returns
        -------
        pd.DataFrame — columns: feature_value, mean_pd.
        """
        from sklearn.inspection import partial_dependence as _pdp

        if not hasattr(self._model, "estimators_"):
            raise RuntimeError("Model has not been fitted yet.")
        if isinstance(feature, str):
            if self.feature_names_ is None:
                raise ValueError("feature_names_ not set; pass an integer index.")
            feat_idx = self.feature_names_.index(feature)
        else:
            feat_idx = int(feature)
        result = _pdp(self._model, self._to_array(X), features=[feat_idx],
                      grid_resolution=grid_resolution)
        feat_name = self.feature_names_[feat_idx] if self.feature_names_ else f"x{feat_idx}"
        return pd.DataFrame({
            "feature_value": result["grid_values"][0],
            "mean_pd": result["average"][0],
        })


# ---------------------------------------------------------------------------
# XGBoostPD
# ---------------------------------------------------------------------------

class XGBoostPD(_TreeMixin):
    """Gradient boosted trees PD model via XGBoost.

    Typically delivers the highest discrimination on tabular credit data.
    Handles missing values natively, supports early stopping, and offers
    a richer set of regularisation parameters than random forests.

    Shares the ``fit`` / ``predict_proba`` / ``feature_importance_summary``
    interface with ``LogisticPD`` and ``RandomForestPD``.

    Parameters
    ----------
    n_estimators : int
        Number of boosting rounds.
    max_depth : int
        Maximum tree depth (lower = less overfitting).
    learning_rate : float
        Step-size shrinkage (eta).
    subsample : float
        Fraction of training rows sampled per round.
    colsample_bytree : float
        Fraction of features sampled per tree.
    scale_pos_weight : float or None
        Weight of the positive (default) class. Automatically set to
        ``n_non_defaults / n_defaults`` from training data if ``None``.
    reg_alpha : float
        L1 regularisation on weights.
    reg_lambda : float
        L2 regularisation on weights.
    early_stopping_rounds : int or None
        Stops training if eval metric does not improve. Requires
        ``eval_set`` to be passed to ``fit()``.
    n_jobs : int
    random_state : int or None

    Examples
    --------
    >>> model = XGBoostPD(n_estimators=500, max_depth=4, learning_rate=0.05)
    >>> model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    >>> model.predict_proba(X_test)
    >>> model.feature_importance_summary()
    >>> model.feature_importance_summary(importance_type="weight")
    >>> model.learning_curve_data()
    >>> model.parameter_summary()
    """

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        scale_pos_weight: float | None = None,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        early_stopping_rounds: int | None = None,
        n_jobs: int = -1,
        random_state: int | None = 42,
    ) -> None:
        if not _XGB_AVAILABLE:
            raise ImportError(
                "xgboost is required for XGBoostPD. "
                "Install it with: pip install xgboost"
            )
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.scale_pos_weight = scale_pos_weight
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.early_stopping_rounds = early_stopping_rounds
        self.n_jobs = n_jobs
        self.random_state = random_state
        self._model: XGBClassifier | None = None
        self._scale_pos_weight_used: float | None = None
        self._evals_result: dict = {}
        self.feature_names_: list[str] | None = None

    def _build_model(self, scale_pos_weight: float) -> "XGBClassifier":
        return XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            scale_pos_weight=scale_pos_weight,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            early_stopping_rounds=self.early_stopping_rounds,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            eval_metric="auc",
            verbosity=0,
        )

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        eval_set: list | None = None,
        verbose: bool = False,
    ) -> "XGBoostPD":
        """Fit the XGBoost PD model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,) — binary default indicator.
        eval_set : list of (X, y) tuples, optional
            Validation sets for early stopping and learning curve logging.
            Typically ``[(X_val, y_val)]``. Required when
            ``early_stopping_rounds`` is set.
        verbose : bool — print XGBoost round-by-round progress.

        Returns
        -------
        self
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
        X_arr = self._to_array(X)
        y_arr = np.asarray(y)

        if self.scale_pos_weight is not None:
            spw = float(self.scale_pos_weight)
        else:
            n_neg = int((y_arr == 0).sum())
            n_pos = int((y_arr == 1).sum())
            spw = n_neg / max(n_pos, 1)
        self._scale_pos_weight_used = spw

        self._model = self._build_model(scale_pos_weight=spw)

        fit_kwargs: dict = {"verbose": verbose}
        if eval_set is not None:
            fit_kwargs["eval_set"] = [
                (self._to_array(Xe), np.asarray(ye)) for Xe, ye in eval_set
            ]

        self._model.fit(X_arr, y_arr, **fit_kwargs)

        if hasattr(self._model, "evals_result_"):
            self._evals_result = self._model.evals_result_

        return self

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Return predicted default probabilities (shape n_samples,)."""
        if self._model is None:
            raise RuntimeError("Model has not been fitted yet.")
        return self._model.predict_proba(self._to_array(X))[:, 1]

    # --- XGBoost-specific ------------------------------------------------------

    def feature_importance_summary(self, importance_type: str = "gain") -> pd.DataFrame:
        """XGBoost feature importances with selectable importance type.

        Parameters
        ----------
        importance_type : str
            One of ``'gain'`` (default; average gain per split — recommended),
            ``'weight'`` (number of times feature used in splits),
            ``'cover'`` (average sample coverage per split),
            ``'total_gain'``, or ``'total_cover'``.

        Returns
        -------
        pd.DataFrame — columns: feature, importance, importance_pct, importance_type.

        Raises
        ------
        RuntimeError if model not fitted.
        ValueError if importance_type is not recognised.
        """
        valid_types = {"gain", "weight", "cover", "total_gain", "total_cover"}
        if importance_type not in valid_types:
            raise ValueError(f"Importance must be one of {valid_types}, got '{importance_type}'.")
        if self._model is None:
            raise RuntimeError("Model has not been fitted yet.")

        scores = self._model.get_booster().get_score(importance_type=importance_type)
        names = self.feature_names_ or [f"x{i}" for i in range(self._model.n_features_in_)]
        importances = np.array(
            [scores.get(f"f{i}", scores.get(n, 0.0)) for i, n in enumerate(names)],
            dtype=float,
        )
        total = importances.sum()
        return (
            pd.DataFrame({
                "feature": names,
                "importance": importances,
                "importance_pct": importances / total * 100 if total > 0 else importances,
                "importance_type": importance_type,
            })
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    def learning_curve_data(self) -> pd.DataFrame:
        """Training/validation AUC per boosting round.

        Only available when an ``eval_set`` was passed to ``fit()``.
        Use to plot train vs validation curves and diagnose overfitting.

        Returns
        -------
        pd.DataFrame — columns: round, set, metric, value.

        Raises
        ------
        RuntimeError if no eval_set was passed or model not fitted.
        """
        if not self._evals_result:
            raise RuntimeError(
                "No learning curve data. Pass eval_set=[(X_val, y_val)] to fit()."
            )
        rows = []
        for set_name, metrics in self._evals_result.items():
            for metric_name, values in metrics.items():
                for rnd, val in enumerate(values):
                    rows.append({"round": rnd + 1, "set": set_name,
                                    "metric": metric_name, "value": val})
        return pd.DataFrame(rows)

    def best_iteration(self) -> int | None:
        """Best boosting round (only set when early stopping is used)."""
        if self._model is None:
            raise RuntimeError("Model has not been fitted yet.")
        return getattr(self._model, "best_iteration", None)

    @property
    def scale_pos_weight_used_(self) -> float | None:
        """The scale_pos_weight used during training (auto-computed if not set)."""
        return self._scale_pos_weight_used

    def parameter_summary(self) -> pd.Series:
        """All hyperparameters used, including auto-computed scale_pos_weight."""
        return pd.Series({
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "scale_pos_weight_used": self._scale_pos_weight_used,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "early_stopping_rounds": self.early_stopping_rounds,
        }, name="xgboost_params")


# ---------------------------------------------------------------------------
# MertonPD
# ---------------------------------------------------------------------------

class MertonPD:
    """Merton (1974) structural model for probability of default.

    Treats equity as a call option on the firm's assets. Derives the implied
    asset value and volatility from observable equity data iteratively.

    Parameters
    ----------
    risk_free_rate : float — annualised risk-free rate (e.g. 0.05).
    time_horizon : float — horizon in years (e.g. 1.0).
    tol : float — convergence tolerance for the iterative solver.
    max_iter : int — maximum solver iterations.

    References
    ----------
    Merton, R.C. (1974). On the Pricing of Corporate Debt.
    Journal of Finance, 29(2), 449-470.

    Examples
    --------
    >>> model = MertonPD(risk_free_rate=0.05)
    >>> model.estimate_pd(equity_value=50.0, equity_volatility=0.30, debt_face_value=80.0)
    """

    def __init__(
        self,
        risk_free_rate: float = 0.05,
        time_horizon: float = 1.0,
        tol: float = 1e-6,
        max_iter: int = 1000,
    ) -> None:
        self.risk_free_rate = risk_free_rate
        self.time_horizon = time_horizon
        self.tol = tol
        self.max_iter = max_iter

    def _black_scholes_call(self, asset_value: float, debt: float, sigma: float):
        r, T = self.risk_free_rate, self.time_horizon
        d1 = (np.log(asset_value / debt) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        call = asset_value * stats.norm.cdf(d1) - debt * np.exp(-r * T) * stats.norm.cdf(d2)
        return call, stats.norm.cdf(d1)

    def estimate_pd(
        self,
        equity_value: float,
        equity_volatility: float,
        debt_face_value: float,
    ) -> dict:
        """Estimate PD under the Merton model.

        Parameters
        ----------
        equity_value : float — market capitalisation.
        equity_volatility : float — annualised equity volatility (e.g. 0.30).
        debt_face_value : float — book value of debt.

        Returns
        -------
        dict — keys: pd, asset_value, asset_volatility, distance_to_default.
        """
        asset_value = equity_value + debt_face_value
        asset_volatility = equity_volatility * equity_value / asset_value

        for _ in range(self.max_iter):
            _, delta = self._black_scholes_call(asset_value, debt_face_value, asset_volatility)
            new_av = equity_value + (debt_face_value*np.exp(-self.risk_free_rate*self.time_horizon))
            new_sv = equity_volatility * equity_value / (delta * asset_value)
            if abs(new_av - asset_value) < self.tol and abs(new_sv - asset_volatility) < self.tol:
                break
            asset_value, asset_volatility = new_av, new_sv

        r, T = self.risk_free_rate, self.time_horizon
        d2 = (np.log(asset_value / debt_face_value) + (r - 0.5 * asset_volatility**2) * T) / (
            asset_volatility * np.sqrt(T)
        )
        return {
            "pd": float(stats.norm.cdf(-d2)),
            "asset_value": asset_value,
            "asset_volatility": asset_volatility,
            "distance_to_default": d2,
        }
