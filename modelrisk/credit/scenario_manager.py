"""IFRS 9 Scenario Manager — high-level orchestration for non-technical users.

The ``ScenarioManager`` is the primary entry point for running IFRS 9
ECL calculations across multiple macroeconomic scenarios. It hides the
underlying pipeline complexity behind a simple, English-language API.

Typical workflow
----------------
1. Define scenarios (base, downside, upside) with weights and macro paths.
2. Attach a fitted PD model and portfolio data.
3. Call ``run_all()`` to compute per-scenario ECLs.
4. Call ``weighted_ecl()`` for the probability-weighted IFRS 9 ECL.
5. Call ``summary_report()`` for an audit-ready summary DataFrame.

No-code entry point
-------------------
Users uncomfortable with Python can define scenarios entirely in a YAML
file and load via ``ScenarioManager.from_yaml(path)``.

YAML format example::

    discount_rate: 0.05
    scenarios:
        base:
            weight: 0.50
            pd_scalar: 1.0
            label: Central
        downside:
            weight: 0.30
            pd_scalar: 1.8
            label: Adverse
        upside:
            weight: 0.20
            pd_scalar: 0.7
            label: Benign

Examples
--------
>>> mgr = ScenarioManager(discount_rate=0.05)
>>> mgr.add_scenario("base",     weight=0.50, pd_scalar=1.0,  label="Central")
>>> mgr.add_scenario("downside", weight=0.30, pd_scalar=1.8,  label="Adverse")
>>> mgr.add_scenario("upside",   weight=0.20, pd_scalar=0.70, label="Benign")
>>> mgr.attach_portfolio(pd=df['pd_12m'], lgd=df['lgd'], ead=df['ead'],
...                      stage=df['stage'], lifetime_pd=df['lifetime_pd'])
>>> results = mgr.run_all()
>>> print(mgr.weighted_ecl())
>>> print(mgr.summary_report())
>>> mgr.to_yaml("scenarios_q4_2024.yaml")
"""

from __future__ import annotations

# import warnings --- IGNORE ---
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from modelrisk.credit.ifrs9.ecl import ECLCalculator


# ---------------------------------------------------------------------------
# Scenario dataclass
# ---------------------------------------------------------------------------

@dataclass
class Scenario:
    """A single IFRS 9 macroeconomic scenario.

    Parameters
    ----------
    name : str
        Internal identifier (e.g. ``'base'``, ``'downside'``).
    weight : float
        Probability weight. All scenario weights must sum to 1.0.
    pd_scalar : float
        Multiplier applied to base PIT PDs under this scenario.
        ``1.0`` = no adjustment (base case).
        ``> 1.0`` = worse conditions (e.g. 1.8 = 80% higher default rate).
        ``< 1.0`` = better conditions (e.g. 0.7 = 30% lower default rate).
    label : str
        Human-readable label for reports (e.g. ``"Central"``, ``"Adverse"``).
    macro_paths : dict
        Optional macro variable paths — ``{variable_name: value}``
        (e.g. ``{'gdp_growth': -2.5, 'unemployment': 8.0}``).
        Used when a ``MacroOverlay`` is attached.
    description : str
        Free-text description for documentation and audit trail.
    """

    name: str
    weight: float
    pd_scalar: float = 1.0
    label: str = ""
    macro_paths: dict[str, float] = field(default_factory=dict)
    description: str = ""

    def __post_init__(self) -> None:
        if not 0 < self.weight <= 1:
            raise ValueError(
                f"Scenario '{self.name}': weight must be in (0, 1], got {self.weight}."
            )
        if self.pd_scalar <= 0:
            raise ValueError(
                f"Scenario '{self.name}': pd_scalar must be positive, got {self.pd_scalar}."
            )
        if not self.label:
            self.label = self.name.capitalize()


# ---------------------------------------------------------------------------
# ScenarioManager
# ---------------------------------------------------------------------------

class ScenarioManager:
    """Orchestrate IFRS 9 ECL calculations across multiple macro scenarios.

    This is the high-level, business-friendly entry point. It wraps the
    full IFRS 9 pipeline (PIT calibration → staging → ECL) and runs it
    across all configured scenarios, returning probability-weighted results
    and audit-ready reports.

    Parameters
    ----------
    discount_rate : float
        Annual discount rate for ECL discounting (e.g. 0.05 for 5%).
    period_type : str
        ``'monthly'`` or ``'annual'``.
    validate_weights : bool
        If True, raises an error if scenario weights do not sum to 1.0.
        Set to False during interactive scenario design.

    Examples
    --------
    >>> mgr = ScenarioManager(discount_rate=0.05)
    >>> mgr.add_scenario("base",     weight=0.50, pd_scalar=1.0)
    >>> mgr.add_scenario("downside", weight=0.30, pd_scalar=1.8)
    >>> mgr.add_scenario("upside",   weight=0.20, pd_scalar=0.7)
    >>> mgr.attach_portfolio(pd=df.pd_12m, lgd=df.lgd, ead=df.ead,
    ...                      stage=df.stage, lifetime_pd=df.lifetime_pd)
    >>> results = mgr.run_all()
    >>> mgr.weighted_ecl()
    >>> mgr.summary_report()
    """

    def __init__(
        self,
        discount_rate: float = 0.05,
        period_type: str = "monthly",
        validate_weights: bool = True,
    ) -> None:
        self.discount_rate = discount_rate
        self.period_type = period_type
        self.validate_weights = validate_weights

        self._scenarios: dict[str, Scenario] = {}
        self._portfolio: dict[str, np.ndarray] = {}
        self._results: dict[str, pd.DataFrame] = {}
        self._ecl_totals: dict[str, float] = {}

        self._ecl_calc = ECLCalculator(
            discount_rate=discount_rate,
            period_type=period_type,
        )

    # ------------------------------------------------------------------
    # Scenario configuration
    # ------------------------------------------------------------------

    def add_scenario(
        self,
        name: str,
        weight: float,
        pd_scalar: float = 1.0,
        label: str = "",
        macro_paths: dict[str, float] | None = None,
        description: str = "",
    ) -> ScenarioManager:
        """Add or replace a macroeconomic scenario.

        Parameters
        ----------
        name : str
            Unique scenario identifier (e.g. ``'base'``, ``'downside'``).
        weight : float
            Probability weight for this scenario (e.g. 0.50 for 50%).
        pd_scalar : float
            PD multiplier under this scenario. ``1.0`` = base, ``1.8`` =
            80% higher default rates, ``0.7`` = 30% lower default rates.
        label : str
            Human-readable name for reports. Defaults to capitalised name.
        macro_paths : dict, optional
            Macro variable values for this scenario. Only required when
            using a ``MacroOverlay``; ignored otherwise.
        description : str
            Free-text description for the audit trail.

        Returns
        -------
        self — supports method chaining.

        Examples
        --------
        >>> mgr.add_scenario("downside", weight=0.30, pd_scalar=1.8,
        ...                  label="2008-style stress",
        ...                  description="Severe recession, unemployment peaks at 10%")
        """
        self._scenarios[name] = Scenario(
            name=name,
            weight=weight,
            pd_scalar=pd_scalar,
            label=label,
            macro_paths=macro_paths or {},
            description=description,
        )
        # Clear cached results when scenarios change
        self._results = {}
        self._ecl_totals = {}
        return self

    def remove_scenario(self, name: str) -> ScenarioManager:
        """Remove a scenario by name."""
        if name not in self._scenarios:
            raise KeyError(f"Scenario '{name}' not found.")
        del self._scenarios[name]
        self._results = {}
        self._ecl_totals = {}
        return self

    def list_scenarios(self) -> pd.DataFrame:
        """Return a summary of all configured scenarios.

        Returns
        -------
        pd.DataFrame — columns: name, label, weight, pd_scalar, description.
        """
        rows = [
            {
                "name": s.name,
                "label": s.label,
                "weight": s.weight,
                "pd_scalar": s.pd_scalar,
                "description": s.description,
            }
            for s in self._scenarios.values()
        ]
        df = pd.DataFrame(rows)
        if not df.empty:
            df["weight_pct"] = (df["weight"] * 100).round(1).astype(str) + "%"
        return df

    def _check_weights(self) -> None:
        """Raise if weights do not sum to 1.0 (within tolerance)."""
        total = sum(s.weight for s in self._scenarios.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"Scenario weights sum to {total:.4f} — must sum to 1.0. "
                f"Current scenarios: {self.list_scenarios()[['name','weight']].to_dict('records')}"
            )

    # ------------------------------------------------------------------
    # Portfolio attachment
    # ------------------------------------------------------------------

    def attach_portfolio(
        self,
        pd: pd.Series | np.ndarray,
        lgd: pd.Series | np.ndarray,
        ead: pd.Series | np.ndarray,
        stage: pd.Series | np.ndarray | None = None,
        lifetime_pd: pd.Series | np.ndarray | None = None,
        remaining_periods: pd.Series | np.ndarray | None = None,
        exposure_id: pd.Series | np.ndarray | None = None,
    ) -> ScenarioManager:
        """Attach portfolio data for ECL calculation.

        Parameters
        ----------
        pd : array-like of shape (n_exposures,)
            12-month point-in-time PD per exposure (base, pre-scenario).
        lgd : array-like of shape (n_exposures,)
            LGD per exposure (0 to 1).
        ead : array-like of shape (n_exposures,)
            Exposure at default per exposure (monetary).
        stage : array-like of shape (n_exposures,), optional
            IFRS 9 stage assignment (1, 2, or 3). If not provided, all
            exposures are treated as Stage 1 (12-month ECL).
        lifetime_pd : array-like of shape (n_exposures,), optional
            Pre-computed lifetime PD for Stage 2/3 exposures. If not
            provided, a simplified 5× 12-month PD approximation is used.
        remaining_periods : array-like, optional
            Remaining contractual life in periods.
        exposure_id : array-like, optional
            Identifier per exposure for result traceability.

        Returns
        -------
        self
        """
        n = len(pd)
        self._portfolio = {
            "pd": np.asarray(pd, dtype=float),
            "lgd": np.asarray(lgd, dtype=float),
            "ead": np.asarray(ead, dtype=float),
            "stage": (
                np.asarray(stage, dtype=int) if stage is not None
                else np.ones(n, dtype=int)
            ),
            "lifetime_pd": (
                np.asarray(lifetime_pd, dtype=float) if lifetime_pd is not None
                else np.clip(np.asarray(pd, dtype=float) * 5, 0, 0.9999)
            ),
            "remaining_periods": (
                np.asarray(remaining_periods, dtype=float) if remaining_periods is not None
                else None
            ),
            "exposure_id": (
                np.asarray(exposure_id) if exposure_id is not None
                else np.arange(n)
            ),
        }
        self._results = {}
        self._ecl_totals = {}
        return self

    def _require_portfolio(self) -> None:
        if not self._portfolio:
            raise RuntimeError(
                "No portfolio attached. Call attach_portfolio() first."
            )

    def _require_scenarios(self) -> None:
        if not self._scenarios:
            raise RuntimeError(
                "No scenarios defined. Call add_scenario() first."
            )

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run_scenario(self, name: str) -> pd.DataFrame:
        """Run ECL calculation for a single named scenario.

        Applies the scenario's ``pd_scalar`` to the base PDs, then
        passes through the ECL calculator.

        Parameters
        ----------
        name : str — scenario identifier.

        Returns
        -------
        pd.DataFrame — exposure-level ECL results for this scenario.
        """
        self._require_portfolio()
        if name not in self._scenarios:
            raise KeyError(f"Scenario '{name}' not found. Available: {list(self._scenarios)}")

        scen = self._scenarios[name]
        p = self._portfolio

        # Apply scenario PD scalar (clipped to [0, 1])
        scenario_pd = np.clip(p["pd"] * scen.pd_scalar, 0.0001, 0.9999)
        scenario_lifetime_pd = np.clip(
            p["lifetime_pd"] * scen.pd_scalar, 0.0001, 0.9999
        )

        result = self._ecl_calc.compute_portfolio(
            pd_array=scenario_pd,
            lgd_array=p["lgd"],
            ead_array=p["ead"],
            stage_array=p["stage"],
            lifetime_pd_array=scenario_lifetime_pd,
            remaining_periods_array=p.get("remaining_periods"),
        )
        result.insert(0, "exposure_id", p["exposure_id"])
        result.insert(1, "scenario", name)
        result.insert(2, "scenario_label", scen.label)
        result.insert(3, "pd_scalar", scen.pd_scalar)

        self._results[name] = result
        self._ecl_totals[name] = float(result["ecl"].sum())
        return result

    def run_all(self) -> dict[str, pd.DataFrame]:
        """Run ECL calculation for all configured scenarios.

        Returns
        -------
        dict mapping scenario name → exposure-level ECL DataFrame.

        Raises
        ------
        RuntimeError if no portfolio or scenarios are configured.
        ValueError if weights do not sum to 1.0 and validate_weights=True.
        """
        self._require_portfolio()
        self._require_scenarios()
        if self.validate_weights:
            self._check_weights()

        for name in self._scenarios:
            self.run_scenario(name)

        return dict(self._results)

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def weighted_ecl(self) -> float:
        """Probability-weighted ECL across all scenarios.

        Must call ``run_all()`` (or ``run_scenario()`` for each scenario)
        before calling this.

        Returns
        -------
        float — IFRS 9 probability-weighted ECL (monetary units).
        """
        if not self._ecl_totals:
            raise RuntimeError("Run run_all() before calling weighted_ecl().")
        return float(sum(
            self._ecl_totals[name] * self._scenarios[name].weight
            for name in self._ecl_totals
        ))

    def scenario_ecl_table(self) -> pd.DataFrame:
        """Per-scenario ECL totals with weights and weighted contribution.

        Returns
        -------
        pd.DataFrame — columns: scenario, label, weight, total_ecl,
            weighted_ecl_contribution, coverage_ratio.
        """
        if not self._ecl_totals:
            raise RuntimeError("Run run_all() before calling scenario_ecl_table().")
        total_ead = self._portfolio["ead"].sum()
        rows = []
        for name, ecl in self._ecl_totals.items():
            s = self._scenarios[name]
            rows.append({
                "scenario": name,
                "label": s.label,
                "weight": s.weight,
                "pd_scalar": s.pd_scalar,
                "total_ecl": round(ecl, 2),
                "weighted_contribution": round(ecl * s.weight, 2),
                "coverage_ratio_pct": round(ecl / total_ead * 100, 4) if total_ead > 0 else 0.0,
            })
        df = pd.DataFrame(rows)

        # Grand total row
        total_row = pd.DataFrame([{
            "scenario": "WEIGHTED TOTAL",
            "label": "Probability-weighted ECL",
            "weight": 1.0,
            "pd_scalar": np.nan,
            "total_ecl": np.nan,
            "weighted_contribution": round(self.weighted_ecl(), 2),
            "coverage_ratio_pct": round(
                self.weighted_ecl() / total_ead * 100, 4
            ) if total_ead > 0 else 0.0,
        }])
        return pd.concat([df, total_row], ignore_index=True)

    def summary_report(self) -> pd.DataFrame:
        """Audit-ready summary report combining all scenario results.

        Returns
        -------
        pd.DataFrame — per-stage breakdown across all scenarios plus
            weighted ECL. Suitable for disclosure or management reporting.
        """
        if not self._results:
            raise RuntimeError("Run run_all() before calling summary_report().")

        rows = []
        for name, result_df in self._results.items():
            scen = self._scenarios[name]
            stage_summary = self._ecl_calc.summary(result_df)
            for _, row in stage_summary.iterrows():
                rows.append({
                    "scenario": name,
                    "label": scen.label,
                    "weight": scen.weight,
                    "pd_scalar": scen.pd_scalar,
                    "stage": row["stage"],
                    "n_exposures": row["n_exposures"],
                    "total_ead": round(row["total_ead"], 2),
                    "total_ecl": round(row["total_ecl"], 2),
                    "coverage_ratio_pct": round(row["coverage_ratio"] * 100, 4),
                    "mean_pd_used": round(row["mean_pd_used"], 6),
                    "weighted_ecl": round(row["total_ecl"] * scen.weight, 2),
                })

        df = pd.DataFrame(rows)

        # Append weighted total row
        weighted_total = pd.DataFrame([{
            "scenario": "WEIGHTED TOTAL",
            "label": "Probability-weighted",
            "weight": 1.0,
            "pd_scalar": np.nan,
            "stage": "ALL",
            "n_exposures": len(self._portfolio["ead"]),
            "total_ead": round(self._portfolio["ead"].sum(), 2),
            "total_ecl": np.nan,
            "coverage_ratio_pct": np.nan,
            "mean_pd_used": np.nan,
            "weighted_ecl": round(self.weighted_ecl(), 2),
        }])
        return pd.concat([df, weighted_total], ignore_index=True)

    def exposure_level_results(self, scenario: str | None = None) -> pd.DataFrame:
        """Return exposure-level results, optionally filtered to one scenario.

        Parameters
        ----------
        scenario : str or None
            Scenario name to filter to. If None, all scenarios are returned.

        Returns
        -------
        pd.DataFrame — exposure × scenario results stacked vertically.
        """
        if not self._results:
            raise RuntimeError("Run run_all() first.")
        if scenario is not None:
            if scenario not in self._results:
                raise KeyError(f"No results for scenario '{scenario}'.")
            return self._results[scenario].copy()
        return pd.concat(list(self._results.values()), ignore_index=True)

    # ------------------------------------------------------------------
    # Serialisation — YAML (no-code config)
    # ------------------------------------------------------------------

    def to_yaml(self, path: str | Path) -> None:
        """Export scenario configuration to a YAML file.

        Allows non-technical users to edit scenario weights and scalars
        in a text editor, then reload with ``from_yaml()``.

        Parameters
        ----------
        path : str or Path — output file path.
        """
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "PyYAML is required for YAML export. pip install pyyaml"
            )

        config: dict[str, Any] = {
            "discount_rate": self.discount_rate,
            "period_type": self.period_type,
            "scenarios": {},
        }
        for name, scen in self._scenarios.items():
            config["scenarios"][name] = {
                "weight": scen.weight,
                "pd_scalar": scen.pd_scalar,
                "label": scen.label,
                "description": scen.description,
                "macro_paths": scen.macro_paths,
            }

        with open(path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str | Path) -> ScenarioManager:
        """Load scenario configuration from a YAML file.

        No-code entry point — users define scenarios in YAML and run
        the pipeline without writing Python.

        Parameters
        ----------
        path : str or Path — path to YAML config file.

        Returns
        -------
        ScenarioManager — configured and ready for ``attach_portfolio()``
        and ``run_all()``.

        Example YAML
        ------------
        .. code-block:: yaml

            discount_rate: 0.05
            period_type: monthly
            scenarios:
            base:
                weight: 0.50
                pd_scalar: 1.0
                label: Central
                description: Base economic outlook
            downside:
                weight: 0.30
                pd_scalar: 1.8
                label: Adverse
                description: Severe recession scenario
            upside:
                weight: 0.20
                pd_scalar: 0.7
                label: Benign
                description: Rapid recovery scenario
        """
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "PyYAML is required for YAML loading. pip install pyyaml"
            )

        with open(path) as f:
            config = yaml.safe_load(f)

        mgr = cls(
            discount_rate=config.get("discount_rate", 0.05),
            period_type=config.get("period_type", "monthly"),
        )
        for name, scen_cfg in config.get("scenarios", {}).items():
            mgr.add_scenario(
                name=name,
                weight=scen_cfg["weight"],
                pd_scalar=scen_cfg.get("pd_scalar", 1.0),
                label=scen_cfg.get("label", ""),
                macro_paths=scen_cfg.get("macro_paths", {}),
                description=scen_cfg.get("description", ""),
            )
        return mgr

    @classmethod
    def from_dict(cls, config: dict) -> ScenarioManager:
        """Load scenario configuration from a Python dictionary.

        Useful when scenarios are defined programmatically or loaded
        from a database / API rather than a file.

        Parameters
        ----------
        config : dict — same structure as the YAML format.

        Returns
        -------
        ScenarioManager
        """
        mgr = cls(
            discount_rate=config.get("discount_rate", 0.05),
            period_type=config.get("period_type", "monthly"),
        )
        for name, scen_cfg in config.get("scenarios", {}).items():
            mgr.add_scenario(
                name=name,
                weight=scen_cfg["weight"],
                pd_scalar=scen_cfg.get("pd_scalar", 1.0),
                label=scen_cfg.get("label", ""),
                macro_paths=scen_cfg.get("macro_paths", {}),
                description=scen_cfg.get("description", ""),
            )
        return mgr

    def __repr__(self) -> str:
        n_scen = len(self._scenarios)
        n_exp = len(self._portfolio.get("ead", []))
        return (
            f"ScenarioManager("
            f"scenarios={n_scen}, "
            f"exposures={n_exp:,}, "
            f"discount_rate={self.discount_rate:.1%})"
        )
