"""Model explainability: SHAP and LIME with lightweight fallbacks.

If the optional ``shap`` and ``lime`` packages are installed
(``pip install modelrisk[explainability]``), this module wraps them
directly. When they are not available, lightweight from-scratch
implementations are used so the package remains functional without
the optional dependencies.
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

try:
    import lime.lime_tabular as _lime_tabular
    _LIME_AVAILABLE = True
except ImportError:
    _LIME_AVAILABLE = False


# ---------------------------------------------------------------------------
# Lightweight fallback implementations
# ---------------------------------------------------------------------------

class _KernelSHAPFallback:
    """Minimal Kernel SHAP approximation (fallback when shap is not installed).

    Uses a simplified sampling approach: for each feature, the marginal
    contribution is estimated by toggling the feature on/off across a set
    of background samples. This is not a full SHAP implementation but
    provides directionally correct attributions for model validation.

    Parameters
    ----------
    predict_fn : callable
        Function mapping (n_samples, n_features) → probabilities.
    background : np.ndarray
        Background dataset used to marginalise features.
    n_samples : int
        Number of coalition samples per feature.
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

    def shap_values(self, X: np.ndarray) -> np.ndarray:
        """Compute approximate SHAP values for each instance in X.

        Parameters
        ----------
        X : np.ndarray of shape (n_instances, n_features)

        Returns
        -------
        np.ndarray of shape (n_instances, n_features)
        """
        X = np.atleast_2d(X)
        n_instances, n_features = X.shape
        shap_vals = np.zeros((n_instances, n_features))

        bg_idx = self.rng.integers(0, len(self.background), size=self.n_samples)
        bg_samples = self.background[bg_idx]

        for i, x in enumerate(X):
            baseline_pred = float(np.mean(self.predict_fn(bg_samples)))
            for j in range(n_features):
                # With feature j from x
                with_j = bg_samples.copy()
                with_j[:, j] = x[j]
                pred_with = float(np.mean(self.predict_fn(with_j)))

                # Without feature j (marginalised)
                without_j = bg_samples.copy()
                pred_without = float(np.mean(self.predict_fn(without_j)))

                shap_vals[i, j] = pred_with - pred_without

            # Normalise so shap values sum to f(x) - E[f(x)]
            instance_pred = float(self.predict_fn(x.reshape(1, -1))[0])
            total = shap_vals[i].sum()
            if abs(total) > 1e-9:
                shap_vals[i] *= (instance_pred - baseline_pred) / total

        return shap_vals


class _LIMEFallback:
    """Minimal LIME-style local linear approximation (fallback).

    Perturbs features around a query instance, weights samples by proximity,
    and fits a weighted ridge regression to approximate local feature
    importance.

    Parameters
    ----------
    predict_fn : callable
    feature_names : list of str
    n_samples : int
    kernel_width : float
        Controls locality — smaller values = tighter neighbourhood.
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
        self, x: np.ndarray, training_data: np.ndarray
    ) -> dict[str, float]:
        """Explain a single instance with local linear attribution.

        Parameters
        ----------
        x : np.ndarray of shape (n_features,)
        training_data : np.ndarray — used to estimate feature std for perturbation.

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

        # Generate perturbed samples
        noise = self.rng.normal(0, 1, size=(self.n_samples, n_features))
        perturbed = x + noise * stds

        # Kernel weights (exponential based on normalised distance)
        dists = np.sqrt(np.sum(noise**2, axis=1))
        weights = np.exp(-(dists**2) / (2 * self.kernel_width**2 * n_features))

        # Predict on perturbed samples
        preds = self.predict_fn(perturbed)

        # Fit weighted ridge regression
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
    """Unified model explainability interface supporting SHAP and LIME.

    Automatically uses installed ``shap`` / ``lime`` libraries when available,
    falling back to lightweight built-in implementations otherwise.

    Parameters
    ----------
    model : fitted model object
        Must expose a ``predict_proba`` method returning shape (n, 2),
        or a ``predict`` method returning shape (n,).
    feature_names : list of str, optional
        Column names for display in outputs.
    background_data : array-like, optional
        Background dataset for SHAP baseline (recommended: a sample of the
        training set, typically 50–200 rows).
    random_state : int or None

    Examples
    --------
    >>> explainer = Explainer(model, feature_names=X.columns.tolist(), background_data=X_train)
    >>> shap_df = explainer.shap_values(X_test)
    >>> lime_df = explainer.lime_explain(X_test.iloc[0], X_train)
    >>> explainer.feature_importance_summary(X_test)
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
            if isinstance(background_data, pd.DataFrame):
                self._background = background_data.values.astype(float)
            else:
                self._background = np.asarray(background_data, dtype=float)
        else:
            self._background = None

        self._predict_fn = self._build_predict_fn()

    def _build_predict_fn(self) -> Callable:
        """Return a unified predict function → 1D probability array."""
        if hasattr(self.model, "predict_proba"):
            def _proba(X: np.ndarray) -> np.ndarray:
                out = self.model.predict_proba(np.asarray(X))
                # handle both (n, 2) sklearn-style and (n,) custom models
                if out.ndim == 2:
                    return out[:, 1]
                return out
            return _proba
        elif hasattr(self.model, "predict"):
            return lambda X: np.asarray(self.model.predict(np.asarray(X)), dtype=float)
        else:
            raise AttributeError(
                "Model must have a predict_proba or predict method."
            )

    # ------------------------------------------------------------------
    # SHAP
    # ------------------------------------------------------------------

    def shap_values(
        self,
        X: pd.DataFrame | np.ndarray,
        n_background_samples: int = 100,
    ) -> pd.DataFrame:
        """Compute SHAP values for all instances in X.

        Uses the ``shap`` library (TreeExplainer for tree models,
        KernelExplainer otherwise) when available; falls back to the
        built-in approximate kernel SHAP.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        n_background_samples : int
            Rows of background to summarise with shap.kmeans when using
            the shap library.

        Returns
        -------
        pd.DataFrame of shape (n_samples, n_features) with SHAP values.
        """
        X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=float)
        names = self.feature_names or [f"feature_{i}" for i in range(X_arr.shape[1])]
        background = self._background if self._background is not None else X_arr

        if _SHAP_AVAILABLE:
            return self._shap_library(X_arr, background, names, n_background_samples)
        else:
            warnings.warn(
                "shap not installed. Using built-in fallback. "
                "Install with: pip install modelrisk[explainability]",
                ImportWarning,
                stacklevel=2,
            )
            return self._shap_fallback(X_arr, background, names)

    def _shap_library(
        self,
        X_arr: np.ndarray,
        background: np.ndarray,
        names: list[str],
        n_background_samples: int,
    ) -> pd.DataFrame:
        """SHAP via the shap library — TreeExplainer or KernelExplainer."""
        try:
            explainer = _shap.TreeExplainer(self.model)
            vals = explainer.shap_values(X_arr)
            if isinstance(vals, list):
                vals = vals[1]
        except Exception:
            summary = _shap.kmeans(background, min(n_background_samples, len(background)))
            explainer = _shap.KernelExplainer(self._predict_fn, summary)
            vals = explainer.shap_values(X_arr, silent=True)
        return pd.DataFrame(vals, columns=names)

    def _shap_fallback(
        self, X_arr: np.ndarray, background: np.ndarray, names: list[str]
    ) -> pd.DataFrame:
        fallback = _KernelSHAPFallback(
            self._predict_fn, background, random_state=self.random_state
        )
        vals = fallback.shap_values(X_arr)
        return pd.DataFrame(vals, columns=names)

    # ------------------------------------------------------------------
    # LIME
    # ------------------------------------------------------------------

    def lime_explain(
        self,
        instance: pd.Series | np.ndarray,
        training_data: pd.DataFrame | np.ndarray,
        n_samples: int = 500,
        top_n: int | None = None,
    ) -> pd.DataFrame:
        """LIME explanation for a single instance.

        Uses the ``lime`` library when available; falls back to the
        built-in local linear approximation.

        Parameters
        ----------
        instance : array-like of shape (n_features,)
            The observation to explain.
        training_data : array-like of shape (n_train, n_features)
            Training data for feature statistics.
        n_samples : int
            Number of perturbed samples to generate.
        top_n : int or None
            Return only the top_n most important features by magnitude.

        Returns
        -------
        pd.DataFrame with columns: feature, importance, abs_importance.
            Sorted by absolute importance descending.
        """
        x_arr = instance.values if isinstance(instance, pd.Series) else np.asarray(instance, dtype=float)
        train_arr = (
            training_data.values if isinstance(training_data, pd.DataFrame)
            else np.asarray(training_data, dtype=float)
        )
        names = self.feature_names or [f"feature_{i}" for i in range(len(x_arr))]

        if _LIME_AVAILABLE:
            importance = self._lime_library(x_arr, train_arr, names, n_samples)
        else:
            warnings.warn(
                "lime not installed. Using built-in fallback. "
                "Install with: pip install modelrisk[explainability]",
                ImportWarning,
                stacklevel=2,
            )
            fallback = _LIMEFallback(
                self._predict_fn,
                feature_names=names,
                n_samples=n_samples,
                random_state=self.random_state,
            )
            importance = fallback.explain_instance(x_arr, train_arr)

        df = pd.DataFrame(
            list(importance.items()), columns=["feature", "importance"]
        )
        df["abs_importance"] = df["importance"].abs()
        df = df.sort_values("abs_importance", ascending=False).reset_index(drop=True)

        if top_n is not None:
            df = df.head(top_n)
        return df

    def _lime_library(
        self,
        x_arr: np.ndarray,
        train_arr: np.ndarray,
        names: list[str],
        n_samples: int,
    ) -> dict[str, float]:
        """LIME explanation via the lime library."""
        lime_explainer = _lime_tabular.LimeTabularExplainer(
            training_data=train_arr,
            feature_names=names,
            mode="classification",
            random_state=self.random_state or 42,
        )
        exp = lime_explainer.explain_instance(
            x_arr,
            self._predict_fn,
            num_features=len(names),
            num_samples=n_samples,
        )
        return dict(exp.as_list())

    # ------------------------------------------------------------------
    # Permutation feature importance
    # ------------------------------------------------------------------

    def permutation_importance(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        n_repeats: int = 10,
        metric: str = "auc",
    ) -> pd.DataFrame:
        """Model-agnostic permutation feature importance.

        Measures the drop in model performance when each feature's values
        are randomly shuffled, breaking its relationship with the target.
        No external libraries required.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        n_repeats : int
            Number of permutations per feature.
        metric : str
            Scoring metric: 'auc', 'accuracy', or 'mse'.

        Returns
        -------
        pd.DataFrame with columns: feature, mean_importance, std_importance.
            Sorted by mean_importance descending.
        """
        from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error

        X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=float)
        y_arr = np.asarray(y)
        names = self.feature_names or [f"feature_{i}" for i in range(X_arr.shape[1])]
        rng = np.random.default_rng(self.random_state)

        def _score(X_in: np.ndarray) -> float:
            preds = self._predict_fn(X_in)
            if metric == "auc":
                return float(roc_auc_score(y_arr, preds))
            elif metric == "accuracy":
                return float(accuracy_score(y_arr, (preds >= 0.5).astype(int)))
            elif metric == "mse":
                return -float(mean_squared_error(y_arr, preds))
            raise ValueError(f"Unknown metric '{metric}'. Choose: auc, accuracy, mse.")

        baseline = _score(X_arr)
        results = []

        for j, name in enumerate(names):
            drops = []
            for _ in range(n_repeats):
                X_perm = X_arr.copy()
                X_perm[:, j] = rng.permutation(X_perm[:, j])
                drops.append(baseline - _score(X_perm))
            results.append(
                {
                    "feature": name,
                    "mean_importance": float(np.mean(drops)),
                    "std_importance": float(np.std(drops, ddof=1)),
                }
            )

        return (
            pd.DataFrame(results)
            .sort_values("mean_importance", ascending=False)
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def feature_importance_summary(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray | None = None,
    ) -> pd.DataFrame:
        """Combined feature importance: mean |SHAP| and permutation importance.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like, optional
            If provided, permutation importance is also computed.

        Returns
        -------
        pd.DataFrame sorted by mean |SHAP| descending.
        """
        shap_df = self.shap_values(X)
        mean_shap = shap_df.abs().mean().rename("mean_abs_shap")

        summary = mean_shap.reset_index()
        summary.columns = ["feature", "mean_abs_shap"]
        summary = summary.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

        if y is not None:
            perm = self.permutation_importance(X, y)
            summary = summary.merge(
                perm[["feature", "mean_importance"]].rename(
                    columns={"mean_importance": "permutation_importance"}
                ),
                on="feature",
                how="left",
            )

        return summary

    @property
    def shap_available(self) -> bool:
        """Whether the shap library is installed."""
        return _SHAP_AVAILABLE

    @property
    def lime_available(self) -> bool:
        """Whether the lime library is installed."""
        return _LIME_AVAILABLE
