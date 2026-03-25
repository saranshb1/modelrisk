"""Macroeconomic scenario conditioning for IFRS 9 forward-looking information (FLI).

IFRS 9 paragraph 5.5.17 requires that ECL estimates incorporate
forward-looking information, including macroeconomic forecasts.
The standard approach is to define multiple macro scenarios, assign
probability weights, and compute a probability-weighted ECL.

This module handles the macro-to-PD mapping and scenario weighting.
The higher-level ``ScenarioManager`` is the intended entry point for
most users; this module contains the underlying mechanics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


class MacroOverlay:
    """Apply macroeconomic scenario paths to PIT PD estimates.

    Maps macro variable forecasts (e.g. GDP growth, unemployment rate)
    to PD adjustments using a fitted sensitivity model, then applies
    them to a base PIT PD.

    Two sensitivity approaches:

    ``'linear'``
        PD adjustment is a linear function of macro variable deviations
        from baseline. Transparent and auditable. Most common in practice.

    ``'logit_link'``
        Adjustments are made in log-odds space to keep PDs in [0, 1].
        Theoretically preferable for large deviations from baseline.

    Parameters
    ----------
    method : str — ``'linear'`` or ``'logit_link'``.

    Examples
    --------
    >>> overlay = MacroOverlay(method='logit_link')
    >>> overlay.fit_sensitivity(
    ...     historical_pd=dr_series,
    ...     macro_df=macro_history[['gdp_growth', 'unemployment']],
    ... )
    >>> adjusted_pd = overlay.apply(
    ...     base_pit_pd=0.025,
    ...     scenario_macro={'gdp_growth': -2.5, 'unemployment': 8.0},
    ...     baseline_macro={'gdp_growth': 1.5, 'unemployment': 5.5},
    ... )
    """

    def __init__(self, method: str = "logit_link") -> None:
        if method not in ("linear", "logit_link"):
            raise ValueError("method must be 'linear' or 'logit_link'.")
        self.method = method
        self._coef: pd.Series | None = None
        self._feature_names: list[str] = []

    @staticmethod
    def _logit(p: float | np.ndarray) -> float | np.ndarray:
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return np.log(p / (1 - p))

    @staticmethod
    def _sigmoid(x: float | np.ndarray) -> float | np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    def fit_sensitivity(
        self,
        historical_pd: pd.Series | np.ndarray,
        macro_df: pd.DataFrame,
        lag_periods: int = 0,
    ) -> MacroOverlay:
        """Estimate PD sensitivity to macroeconomic variables via OLS.

        Parameters
        ----------
        historical_pd : array-like of shape (n_periods,)
            Historical observed default rates (or model PDs), one per period.
        macro_df : pd.DataFrame of shape (n_periods, n_macro_vars)
            Macroeconomic variables aligned to the same periods.
        lag_periods : int
            Lag to apply to macro variables (e.g. 1 = macro at t-1 predicts
            PD at t). Useful when macro data leads credit quality.

        Returns
        -------
        self
        """
        from sklearn.linear_model import LinearRegression

        pd_arr = np.asarray(historical_pd, dtype=float)
        macro = macro_df.copy()
        self._feature_names = list(macro.columns)

        if lag_periods > 0:
            macro = macro.shift(lag_periods).dropna()
            pd_arr = pd_arr[lag_periods:]

        if self.method == "logit_link":
            y = self._logit(pd_arr)
        else:
            y = pd_arr

        x_mat = macro.values
        model = LinearRegression(fit_intercept=True)
        model.fit(x_mat, y)
        self._coef = pd.Series(model.coef_, index=self._feature_names)
        self._intercept = float(model.intercept_)
        return self

    def apply(
        self,
        base_pit_pd: float | np.ndarray,
        scenario_macro: dict[str, float],
        baseline_macro: dict[str, float],
    ) -> float | np.ndarray:
        """Apply macro scenario to adjust a base PIT PD.

        Parameters
        ----------
        base_pit_pd : float or array-like
            Base PIT PD(s) before scenario conditioning.
        scenario_macro : dict
            Macro variable values under the scenario
            (e.g. ``{'gdp_growth': -2.5, 'unemployment': 8.0}``).
        baseline_macro : dict
            Macro variable values under the baseline
            (e.g. ``{'gdp_growth': 1.5, 'unemployment': 5.5}``).

        Returns
        -------
        float or np.ndarray — scenario-adjusted PD(s).
        """
        if self._coef is None:
            raise RuntimeError("Call fit_sensitivity() first.")

        # Compute deviation from baseline
        delta = np.array([
            scenario_macro.get(f, 0.0) - baseline_macro.get(f, 0.0)
            for f in self._feature_names
        ])
        adjustment = float(self._coef.values @ delta)

        if self.method == "logit_link":
            base_logit = self._logit(np.asarray(base_pit_pd, dtype=float))
            adjusted = self._sigmoid(base_logit + adjustment)
        else:
            adjusted = np.asarray(base_pit_pd, dtype=float) + adjustment

        return float(np.clip(adjusted, 1e-6, 1 - 1e-6)) if np.isscalar(base_pit_pd) \
            else np.clip(adjusted, 1e-6, 1 - 1e-6)

    def sensitivity_summary(self) -> pd.DataFrame:
        """Return fitted macro sensitivities.

        Returns
        -------
        pd.DataFrame — columns: variable, coefficient, interpretation.
        """
        if self._coef is None:
            raise RuntimeError("Call fit_sensitivity() first.")
        rows = []
        for var, coef in self._coef.items():
            direction = "increases" if coef > 0 else "decreases"
            if self.method == "logit_link":
                interp = f"1-unit rise {direction} log-odds of default by {abs(coef):.4f}"
            else:
                interp = f"1-unit rise {direction} PD by {abs(coef):.6f}"
            rows.append({"variable": var, "coefficient": coef, "interpretation": interp})
        return pd.DataFrame(rows)
