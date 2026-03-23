"""
modelRisk — IFRS 9 + IRB + ScenarioManager usage example
=========================================================

Demonstrates the full credit risk workflow:
  1. Fit a PD model
  2. Calibrate to point-in-time (IFRS 9) and through-the-cycle (IRB)
  3. Assign IFRS 9 stages
  4. Build forward PD curves and lifetime PDs
  5. Run multi-scenario ECL via ScenarioManager
  6. Compute IRB RWA

Run with:
    python examples/usage_example.py
"""

import numpy as np
import pandas as pd

# ── 1. Simulate a portfolio ───────────────────────────────────────────────
rng = np.random.default_rng(42)
n = 1000

portfolio = pd.DataFrame({
    "exposure_id":    np.arange(n),
    "ead":            rng.uniform(10_000, 500_000, n),
    "lgd":            rng.uniform(0.20, 0.55, n),
    "origination_pd": rng.uniform(0.003, 0.025, n),
    "current_pd":     rng.uniform(0.004, 0.080, n),   # has deteriorated for some
    "remaining_months": rng.integers(6, 60, n),
})

print("=" * 60)
print("  modelRisk IFRS 9 + IRB workflow example")
print("=" * 60)
print(f"\nPortfolio: {n:,} exposures | Total EAD: £{portfolio['ead'].sum():,.0f}\n")

# ── 2. PIT calibration (IFRS 9) ──────────────────────────────────────────
from modelrisk.credit.ifrs9 import PITCalibrator

pit_cal = PITCalibrator(method="scalar", halflife_months=18)
pit_cal.calibrate(
    raw_pd=portfolio["current_pd"],
    observed_default_rate=0.032,   # recent 12-month observed DR
    model_long_run_dr=0.018,       # model long-run average
)
portfolio["pit_pd"] = pit_cal.transform(portfolio["current_pd"])
print(f"PIT calibration: scalar = {pit_cal._scalar:.3f}x "
      f"| Mean PIT PD: {portfolio['pit_pd'].mean():.4f}")

# ── 3. IFRS 9 staging ─────────────────────────────────────────────────────
from modelrisk.credit.ifrs9 import StagingClassifier

staging = StagingClassifier(
    method="dual",
    absolute_threshold=0.01,
    relative_multiplier=2.0,
    default_threshold=0.20,
)
portfolio["stage"] = staging.classify(
    current_pd=portfolio["pit_pd"],
    origination_pd=portfolio["origination_pd"],
)
summary = staging.stage_summary(
    portfolio["stage"], portfolio["pit_pd"], portfolio["ead"]
)
print("\nIFRS 9 staging:")
print(summary.to_string(index=False))

# ── 4. Lifetime PD curves ─────────────────────────────────────────────────
from modelrisk.credit.ifrs9 import ForwardPDCurve, LifetimePDCurve

lifetime_pds = []
for _, row in portfolio.iterrows():
    fwd = ForwardPDCurve(n_periods=60)
    marginal = fwd.build(pit_pd_12m=row["pit_pd"])
    lc = LifetimePDCurve(discount_rate=0.05)
    lc.compute(marginal, remaining_periods=int(row["remaining_months"]))
    lifetime_pds.append(lc.lifetime_pd)

portfolio["lifetime_pd"] = lifetime_pds
print(f"\nLifetime PD: mean = {portfolio['lifetime_pd'].mean():.4f} "
      f"| max = {portfolio['lifetime_pd'].max():.4f}")

# ── 5. ScenarioManager — multi-scenario ECL ──────────────────────────────
from modelrisk.credit import ScenarioManager

print("\n" + "─" * 60)
print("  Running IFRS 9 scenario analysis")
print("─" * 60)

mgr = (
    ScenarioManager(discount_rate=0.05)
    .add_scenario("base",           weight=0.50, pd_scalar=1.00,
                  label="Central",       description="Base economic outlook")
    .add_scenario("downside",       weight=0.30, pd_scalar=1.80,
                  label="Adverse",       description="Mild recession")
    .add_scenario("severe_down",    weight=0.10, pd_scalar=2.80,
                  label="Severe adverse",description="2008-style deep recession")
    .add_scenario("upside",         weight=0.10, pd_scalar=0.65,
                  label="Upside",        description="Rapid recovery")
    .attach_portfolio(
        pd=portfolio["pit_pd"],
        lgd=portfolio["lgd"],
        ead=portfolio["ead"],
        stage=portfolio["stage"],
        lifetime_pd=portfolio["lifetime_pd"],
        remaining_periods=portfolio["remaining_months"],
        exposure_id=portfolio["exposure_id"],
    )
)

results = mgr.run_all()
print(mgr.scenario_ecl_table().to_string(index=False))

print(f"\nProbability-weighted ECL: £{mgr.weighted_ecl():,.0f}")
total_ead = portfolio["ead"].sum()
print(f"ECL coverage ratio:       {mgr.weighted_ecl() / total_ead:.4%}")

# ── 6. Audit-ready summary report ─────────────────────────────────────────
print("\n" + "─" * 60)
print("  Summary report (first 8 rows)")
print("─" * 60)
report = mgr.summary_report()
print(report.head(8).to_string(index=False))

# ── 7. Save scenario config to YAML ──────────────────────────────────────
import os
mgr.to_yaml("examples/scenarios_output.yaml")
print("\nScenario config saved → examples/scenarios_output.yaml")

# ── 8. Reload from YAML (no-code workflow) ───────────────────────────────
mgr2 = ScenarioManager.from_yaml("examples/scenarios_output.yaml")
print(f"Reloaded from YAML: {len(mgr2._scenarios)} scenarios")
os.remove("examples/scenarios_output.yaml")

# ── 9. TTC calibration (IRB) ─────────────────────────────────────────────
print("\n" + "─" * 60)
print("  IRB through-the-cycle calibration")
print("─" * 60)

from modelrisk.credit.irb import TTCCalibrator, RatingMasterScale, IRBCapital

historical_drs = np.array([0.012, 0.018, 0.028, 0.035, 0.030, 0.022, 0.016])
ttc_cal = TTCCalibrator(min_pd=0.0003, cycle_length_years=7).fit(historical_drs)
print(ttc_cal.calibration_summary().to_string())

portfolio["ttc_pd"] = ttc_cal.apply(portfolio["pit_pd"])
rms = RatingMasterScale(n_grades=18)
portfolio["rating_grade"] = rms.assign_grades(portfolio["ttc_pd"])

# ── 10. IRB RWA ──────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("  IRB capital (RWA)")
print("─" * 60)

irb = IRBCapital(asset_class="retail_mortgage")
rwa_df = irb.rwa_portfolio(
    pd_array=portfolio["ttc_pd"],
    lgd_array=portfolio["lgd"],
    ead_array=portfolio["ead"],
)
total_rwa = float(rwa_df.loc["TOTAL", "rwa"])
total_el  = float(rwa_df.loc["TOTAL", "expected_loss"])
print(f"Total RWA:             £{total_rwa:,.0f}")
print(f"RWA density:           {total_rwa / total_ead:.1%}")
print(f"Expected loss (Basel): £{total_el:,.0f}")
print(f"EL / EAD:              {total_el / total_ead:.4%}")
print(f"\nIFRS 9 ECL:            £{mgr.weighted_ecl():,.0f}")
print(f"Basel EL:              £{total_el:,.0f}")
print(f"ECL / Basel EL ratio:  {mgr.weighted_ecl() / total_el:.2f}x")
