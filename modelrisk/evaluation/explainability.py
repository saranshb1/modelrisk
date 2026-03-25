"""Model explainability: SHAP values, local linear explanations, and permutation importance.

SHAP
----
Uses the ``shap`` library when available (``pip install modelrisk[explainability]``),
falling back to a built-in kernel SHAP approximation when it is not installed.

Local linear explanations (LIME-style)
---------------------------------------
Always uses the built-in ``_LocalExplainer`` implementation — a weighted ridge
regression over perturbed neighbourhood samples.  The ``lime`` library has been
removed as a dependency because its ``LimeTabularExplainer`` raises
``NotImplementedError`` for models that return only positive-class probabilities
(a standard pattern in scikit-learn and modelrisk).  The built-in implementation
is equivalent in concept, dependency-free, and fully compatible with any model
that exposes ``predict_proba`` or ``predict``.

Permutation importance
----------------------
Model-agnostic; no external libraries required.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Optional dependency detection
# ---------------------------------------------------------------------------

try:
    import shap as _shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False


# ---------------------------------------------------------------------------
# Built-in kernel SHAP approximation
# ---------------------------------------------------------------------------

class _KernelSHAPFallback:
    """Minimal kernel SHAP approximation (used when shap is not installed).

    For each feature, the marginal contribution is estimated by toggling the
    feature between its observed value and background samples.  Attributions
    are normalised so they sum to ``f(x) - E[f(x)]``.

    Parameters
    ----------
    predict_fn : callable
        Function mapping (n_samples, n_features) → 1-D probability array.
    background : np.ndarray
        Background dataset used to marginalise features.
    n_samples : int
        Number of background samples drawn per instance.
    random_state : int or None
    """

    def __init__(
        self,
        predict_fn: Callable,
        background: np.ndarray,
        n_samples: int = 100,
        random_state: int | None = 42,
    ) -> None:
        self.predict_fn = predict_fn
        self.background = background
        self.n_samples = n_samples
        self.rng = np.random.default_rng(random_state)

    def shap_values(self, x_ex: np.ndarray) -> np.ndarray:
        """Return approximate SHAP values of shape (n_instances, n_features)."""
        x_mat = np.atleast_2d(x_ex)
        n_instances, n_features = x_mat.shape
        shap_vals = np.zeros((n_instances, n_features))

        bg_idx = self.rng.integers(0, len(self.background), size=self.n_samples)
        bg_samples = self.background[bg_idx]

        for i, x in enumerate(x_mat):
            baseline_pred = float(np.mean(self.predict_fn(bg_samples)))
            for j in range(n_features):
                with_j = bg_samples.copy()
                with_j[:, j] = x[j]
                shap_vals[i, j] = float(np.mean(self.predict_fn(with_j))) - float(
                    np.mean(self.predict_fn(bg_samples))
                )
            # Normalise to f(x) - E[f(x)]
            instance_pred = float(self.predict_fn(x.reshape(1, -1))[0])
            total = shap_vals[i].sum()
            if abs(total) > 1e-9:
                shap_vals[i] *= (instance_pred - baseline_pred) / total

        return shap_vals


# ---------------------------------------------------------------------------
# Built-in local linear explainer (LIME-style)
# ---------------------------------------------------------------------------

class _LocalExplainer:
    """Local linear approximation for single-instance explanations.

    Perturbs features around a query instance, weights the perturbed samples
    by their proximity to the query (exponential kernel), and fits a weighted
    ridge regression to approximate local feature importance.

    This is conceptually equivalent to LIME's tabular explainer but is
    implemented from scratch to avoid the ``lime`` library's restriction
    that predict functions must return full class-probability matrices.

    Parameters
    ----------
    predict_fn : callable
        Function mapping (n_samples, n_features) → 1-D probability array.
        This is the standard modelrisk convention — no 2-D output required.
    feature_names : list of str or None
    n_samples : int
        Number of perturbed neighbourhood samples.
    kernel_width : float
        Controls locality.  Smaller = tighter neighbourhood around the query.
    random_state : int or None
    """

    def __init__(
        self,
        predict_fn: Callable,
        feature_names: list[str] | None = None,
        n_samples: int = 500,
        kernel_width: float = 0.75,
        random_state: int | None = 42,
    ) -> None:
        self.predict_fn = predict_fn
        self.feature_names = feature_names
        self.n_samples = n_samples
        self.kernel_width = kernel_width
        self.rng = np.random.default_rng(random_state)

    def explain_instance(
        self,
        x: np.ndarray,
        training_data: np.ndarray,
    ) -> dict[str, float]:
        """Explain a single instance.

        Parameters
        ----------
        x : np.ndarray of shape (n_features,)
            The instance to explain.
        training_data : np.ndarray of shape (n_train, n_features)
            Used to estimate per-feature standard deviations for perturbation.

        Returns
        -------
        dict mapping feature name → local importance coefficient.
        """
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        x = np.asarray(x, dtype=float)
        n_features = len(x)

        stds = np.std(training_data, axis=0, ddof=1)
        stds = np.where(stds < 1e-9, 1.0, stds)

        # Generate perturbed neighbourhood
        noise = self.rng.normal(0, 1, size=(self.n_samples, n_features))
        perturbed = x + noise * stds

        # Exponential kernel weights based on normalised distance
        dists = np.sqrt(np.sum(noise ** 2, axis=1))
        weights = np.exp(-(dists ** 2) / (2 * self.kernel_width ** 2 * n_features))

        # Predict on perturbed samples — returns 1-D array, no issues
        preds = self.predict_fn(perturbed)

        # Fit weighted ridge regression in standardised feature space
        scaler = StandardScaler()
        perturbed_scaled = scaler.fit_transform(perturbed)
        ridge = Ridge(alpha=1.0)
        ridge.fit(perturbed_scaled, preds, sample_weight=weights)

        names = self.feature_names or [f"feature_{i}" for i in range(n_features)]
        return dict(zip(names, ridge.coef_))


# ---------------------------------------------------------------------------
# Public Explainer class
# ---------------------------------------------------------------------------

class Explainer:
    """Unified model explainability: SHAP, local linear explanations, and permutation importance.

    Parameters
    ----------
    model : fitted model object
        Must expose ``predict_proba`` (returning shape ``(n, 2)``) or
        ``predict`` (returning shape ``(n,)``).
    feature_names : list of str, optional
        Column names used in all output DataFrames.
    background_data : array-like, optional
        Background dataset for SHAP baseline computation.
        Recommended: 50–200 rows sampled from the training set.
    random_state : int or None

    Notes
    -----
    **SHAP** — uses the ``shap`` library (TreeExplainer or KernelExplainer)
    when installed; falls back to the built-in ``_KernelSHAPFallback``
    otherwise.  Install with ``pip install modelrisk[explainability]``.

    **Local linear explanations** — always uses the built-in
    ``_LocalExplainer``.  The ``lime`` library has been removed because its
    tabular explainer requires models to return full class-probability matrices
    (shape ``(n, n_classes)``), which is incompatible with the standard
    modelrisk pattern of returning 1-D positive-class probabilities.

    Examples
    --------
    >>> exp = Explainer(model, feature_names=X.columns.tolist(), background_data=X_train)
    >>> shap_df = exp.shap_values(X_test)
    >>> local_df = exp.local_explain(X_test.iloc[0], X_train, top_n=10)
    >>> perm_df = exp.permutation_importance(X_test, y_test)
    >>> summary = exp.feature_importance_summary(X_test, y_test)
    """

    def __init__(
        self,
        model: Any,
        feature_names: list[str] | None = None,
        background_data: pd.DataFrame | np.ndarray | None = None,
        random_state: int | None = 42,
    ) -> None:
        self.model = model
        self.feature_names = feature_names
        self.random_state = random_state

        if background_data is not None:
            self._background = (
                background_data.values.astype(float)
                if isinstance(background_data, pd.DataFrame)
                else np.asarray(background_data, dtype=float)
            )
        else:
            self._background = None

        self._predict_fn = self._build_predict_fn()

    def _build_predict_fn(self) -> Callable:
        """Return a unified 1-D predict function."""
        if hasattr(self.model, "predict_proba"):
            def _proba(X: np.ndarray) -> np.ndarray:
                out = self.model.predict_proba(np.asarray(X))
                return out[:, 1] if out.ndim == 2 else out
            return _proba
        elif hasattr(self.model, "predict"):
            return lambda X: np.asarray(
                self.model.predict(np.asarray(X)), dtype=float
            )
        raise AttributeError(
            "Model must expose predict_proba or predict."
        )

    # ------------------------------------------------------------------
    # SHAP
    # ------------------------------------------------------------------

    def shap_values(
        self,
        x_ex: pd.DataFrame | np.ndarray,
        n_background_samples: int = 100,
    ) -> pd.DataFrame:
        """Compute SHAP values for all instances in x_ex.

        Uses the ``shap`` library (TreeExplainer → KernelExplainer) when
        installed; falls back to the built-in kernel SHAP approximation.

        Parameters
        ----------
        x_ex : array-like of shape (n_samples, n_features)
        n_background_samples : int
            Rows of background to summarise with ``shap.kmeans`` when using
            the ``shap`` library's KernelExplainer.

        Returns
        -------
        pd.DataFrame of shape (n_samples, n_features) with SHAP values.
        """
        x_arr = x_ex.values if isinstance(x_ex, pd.DataFrame) else np.asarray(x_ex, dtype=float)
        names = self.feature_names or [f"feature_{i}" for i in range(x_arr.shape[1])]
        background = self._background if self._background is not None else x_arr

        if _SHAP_AVAILABLE:
            return self._shap_library(x_arr, background, names, n_background_samples)

        warnings.warn(
            "shap not installed — using built-in kernel SHAP approximation. "
            "For full SHAP support: pip install modelrisk[explainability]",
            ImportWarning,
            stacklevel=2,
        )
        return self._shap_fallback(x_arr, background, names)

    def _shap_library(
        self,
        x_arr: np.ndarray,
        background: np.ndarray,
        names: list[str],
        n_background_samples: int,
    ) -> pd.DataFrame:
        try:
            explainer = _shap.TreeExplainer(self.model)
            vals = explainer.shap_values(x_arr)
            if isinstance(vals, list):
                vals = vals[1]
        except Exception:
            summary = _shap.kmeans(background, min(n_background_samples, len(background)))
            explainer = _shap.KernelExplainer(self._predict_fn, summary)
            vals = explainer.shap_values(x_arr, silent=True)
        return pd.DataFrame(vals, columns=names)

    def _shap_fallback(
        self, x_arr: np.ndarray, background: np.ndarray, names: list[str]
    ) -> pd.DataFrame:
        fallback = _KernelSHAPFallback(
            self._predict_fn, background, random_state=self.random_state
        )
        return pd.DataFrame(fallback.shap_values(x_arr), columns=names)

    # ------------------------------------------------------------------
    # Local linear explanations (replaces LIME)
    # ------------------------------------------------------------------

    def local_explain(
        self,
        instance: pd.Series | np.ndarray,
        training_data: pd.DataFrame | np.ndarray,
        n_samples: int = 500,
        kernel_width: float = 0.75,
        top_n: int | None = None,
    ) -> pd.DataFrame:
        """Local linear explanation for a single instance.

        Fits a weighted ridge regression on a perturbed neighbourhood around
        the instance to approximate which features drove that specific
        prediction.  Conceptually equivalent to LIME but implemented from
        scratch — no external dependency, no class-probability restriction.

        Parameters
        ----------
        instance : array-like of shape (n_features,)
            The observation to explain.
        training_data : array-like of shape (n_train, n_features)
            Used to estimate per-feature standard deviations for perturbation.
        n_samples : int
            Number of perturbed neighbourhood samples.
        kernel_width : float
            Exponential kernel width controlling locality.
        top_n : int or None
            If set, return only the top_n features by absolute importance.

        Returns
        -------
        pd.DataFrame — columns: feature, importance, abs_importance.
            Sorted by absolute importance descending.
        """
        x_arr = (
            instance.values if isinstance(instance, pd.Series)
            else np.asarray(instance, dtype=float)
        )
        train_arr = (
            training_data.values if isinstance(training_data, pd.DataFrame)
            else np.asarray(training_data, dtype=float)
        )
        names = self.feature_names or [f"feature_{i}" for i in range(len(x_arr))]

        local_exp = _LocalExplainer(
            predict_fn=self._predict_fn,
            feature_names=names,
            n_samples=n_samples,
            kernel_width=kernel_width,
            random_state=self.random_state,
        )
        importance = local_exp.explain_instance(x_arr, train_arr)

        df = (
            pd.DataFrame(list(importance.items()), columns=["feature", "importance"])
            .assign(abs_importance=lambda d: d["importance"].abs())
            .sort_values("abs_importance", ascending=False)
            .reset_index(drop=True)
        )
        return df.head(top_n) if top_n is not None else df

    # ------------------------------------------------------------------
    # Permutation feature importance
    # ------------------------------------------------------------------

    def permutation_importance(
        self,
        x_ex: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        n_repeats: int = 10,
        metric: str = "auc",
    ) -> pd.DataFrame:
        """Model-agnostic permutation feature importance.

        Measures the drop in model performance when each feature is randomly
        shuffled.  No external libraries required.

        Parameters
        ----------
        x_ex : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        n_repeats : int
            Number of shuffle repetitions per feature.
        metric : str
            ``'auc'``, ``'accuracy'``, or ``'mse'``.

        Returns
        -------
        pd.DataFrame — columns: feature, mean_importance, std_importance.
            Sorted by mean_importance descending.
        """
        from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score

        x_arr = x_ex.values if isinstance(x_ex, pd.DataFrame) else np.asarray(x_ex, dtype=float)
        y_arr = np.asarray(y)
        names = self.feature_names or [f"feature_{i}" for i in range(x_arr.shape[1])]
        rng = np.random.default_rng(self.random_state)

        valid_metrics = ("auc", "accuracy", "mse")
        if metric not in valid_metrics:
            raise ValueError(f"metric must be one of {valid_metrics}, got '{metric}'.")

        def _score(x_in: np.ndarray) -> float:
            preds = self._predict_fn(x_in)
            if metric == "auc":
                return float(roc_auc_score(y_arr, preds))
            if metric == "accuracy":
                return float(accuracy_score(y_arr, (preds >= 0.5).astype(int)))
            return -float(mean_squared_error(y_arr, preds))

        baseline = _score(x_arr)
        results = []
        for j, name in enumerate(names):
            drops = []
            for _ in range(n_repeats):
                x_perm = x_arr.copy()
                x_perm[:, j] = rng.permutation(x_perm[:, j])
                drops.append(baseline - _score(x_perm))
            results.append({
                "feature": name,
                "mean_importance": float(np.mean(drops)),
                "std_importance": float(np.std(drops, ddof=1)),
            })

        return (
            pd.DataFrame(results)
            .sort_values("mean_importance", ascending=False)
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    # Combined summary
    # ------------------------------------------------------------------

    def feature_importance_summary(
        self,
        x_ex: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray | None = None,
    ) -> pd.DataFrame:
        """Combined feature importance: mean |SHAP| + optional permutation importance.

        Parameters
        ----------
        x_ex : array-like of shape (n_samples, n_features)
        y : array-like, optional
            If provided, permutation importance is computed and merged in.

        Returns
        -------
        pd.DataFrame sorted by mean |SHAP| descending.
        """
        shap_df = self.shap_values(x_ex)
        summary = (
            shap_df.abs().mean()
            .rename("mean_abs_shap")
            .reset_index()
            .rename(columns={"index": "feature"})
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )

        if y is not None:
            perm = self.permutation_importance(x_ex, y)
            summary = summary.merge(
                perm[["feature", "mean_importance"]].rename(
                    columns={"mean_importance": "permutation_importance"}
                ),
                on="feature",
                how="left",
            )

        return summary

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def shap_available(self) -> bool:
        """Whether the ``shap`` library is installed."""
        return _SHAP_AVAILABLE

    @property
    def lime_available(self) -> bool:
        """Always ``False`` — the ``lime`` library has been removed.

        Local explanations are now provided by the built-in
        ``_LocalExplainer`` via ``local_explain()``.
        """
        return False
