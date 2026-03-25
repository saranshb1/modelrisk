# modelrisk

[![CI](https://github.com/saranshb1/modelRisk/actions/workflows/ci.yml/badge.svg)](https://github.com/saranshb1/modelRisk/actions/workflows/ci.yml)
[![PyPI version](https://badge.fury.io/py/modelrisk.svg)](https://badge.fury.io/py/modelrisk)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![codecov](https://codecov.io/gh/saranshb1/modelRisk/branch/main/graph/badge.svg)](https://codecov.io/gh/saranshb1/modelRisk)

**modelrisk** is a Python library for credit, market, and operational risk modelling with a built-in model evaluation and explainability suite.

---

## Architecture

![modelRisk package architecture](docs/architecture.png)

> Full package map — five subpackages (`credit`, `market`, `operational`, `evaluation`, `utils`), the IFRS 9 and IRB regulatory pipelines, the high-level `ScenarioManager`, and all cross-package dependencies.

---

## Features

### Credit Risk
- **PD Models** — Logistic regression PD with WoE encoding; Merton structural model
- **LGD Models** — Beta regression and linear regression for loss given default
- **Scorecards** — Weight of Evidence (WoE), Information Value (IV), and points-based scoring

### Market Risk
- **VaR** — Historical simulation, parametric (Normal and Student-t), Monte Carlo
- **CVaR / Expected Shortfall** — All three methods; Basel III 97.5% ES ready
- **Volatility** — EWMA (RiskMetrics) and GARCH(1,1) by MLE with multi-step forecasting

### Operational Risk
- **Loss Distribution Approach (LDA)** — Frequency/severity fitting + Monte Carlo convolution
- **Scenario Analysis** — Expert-elicited scenarios with Poisson-lognormal simulation
- **Extreme Value Theory** — Generalised Pareto Distribution (GPD) via Peaks Over Threshold

### Model Evaluation
| Category | Metrics |
|---|---|
| **Classification** | AUC-ROC, **Gini**, **KS statistic**, F1, Precision, Recall, Specificity, Balanced Accuracy, MCC, Brier Score, Log Loss, Lift@Decile, CAP curve |
| **Regression** | RMSE, MSE, MAE, R², Adjusted R², MAPE, Median AE, Max Error, Mean Bias |
| **Calibration** | **Hosmer-Lemeshow test**, Reliability diagram, ECE, Portfolio rate ratio, EVA decile table |
| **Explainability** | **SHAP values** (TreeExplainer / KernelExplainer / fallback), **LIME** (tabular / fallback), Permutation importance, Feature importance summary |

---

## Installation

```bash
# Core package
pip install modelrisk

# With SHAP and LIME explainability
pip install "modelrisk[explainability]"

# Full install including dev tools
pip install "modelrisk[all]"
```

---

## Quick Start

### Credit Risk — PD Model

```python
from modelrisk.credit import LogisticPD
from modelrisk.evaluation import ClassificationMetrics

model = LogisticPD(scale_features=True)
model.fit(X_train, y_train)

y_pred = model.predict_proba(X_test)

# Full evaluation suite
metrics = ClassificationMetrics(y_test, y_pred)
print(metrics.summary())
#    metric                value   interpretation
# 0  AUC-ROC              0.823   Discrimination; >0.7 acceptable, >0.8 good
# 1  Gini                 0.646   2*AUC - 1; >0.4 acceptable for credit
# 2  KS statistic         0.512   >0.3 acceptable; measures score separation
# ...
```

### Credit Risk — Scorecard with WoE/IV

```python
from modelrisk.credit import Scorecard

sc = Scorecard(pdo=20, base_score=600)
sc.fit(X_binned, y)           # X_binned: pre-binned with pd.cut / pd.qcut

print(sc.information_value_summary())
scores = sc.score(X_test)    # Integer scorecard points, higher = lower risk
```

### Market Risk — VaR and CVaR

```python
from modelrisk.market import HistoricalVaR, CVaR

# 1-day 99% historical VaR
var = HistoricalVaR(confidence_level=0.99).fit(returns).var()

# 10-day 97.5% CVaR (Basel III ES)
es = CVaR(confidence_level=0.975, method="historical", holding_period=10)
es.fit(returns)
print(es.summary())
```

### Market Risk — GARCH Volatility

```python
from modelrisk.market import GARCHVolatility

garch = GARCHVolatility().fit(returns)
print(garch.parameter_summary())

# 10-day annualised volatility forecast
forecasts = garch.forecast(horizon=10)
```

### Operational Risk — LDA Capital

```python
from modelrisk.operational import LossDistributionApproach

lda = LossDistributionApproach(
    frequency_dist="negative_binomial",
    severity_dist="lognormal",
    n_simulations=200_000,
)
lda.fit(annual_frequencies, individual_losses)
capital = lda.capital_estimate()
print(f"99.9% VaR capital: {capital['var_capital']:,.0f}")
print(f"99.9% CVaR capital: {capital['cvar_capital']:,.0f}")
```

### Model Calibration

```python
from modelrisk.evaluation import CalibrationMetrics

cal = CalibrationMetrics(y_true, y_pred_proba, n_bins=10)

# Hosmer-Lemeshow goodness-of-fit
hl = cal.hosmer_lemeshow()
print(f"HL p-value: {hl['p_value']:.4f} — {hl['interpretation']}")

# Decile-level Expected vs Actual table
print(cal.expected_vs_actual())
```

### Explainability — SHAP and LIME

```python
from modelrisk.evaluation import Explainer

# Works with or without the shap/lime packages installed
explainer = Explainer(
    model,
    feature_names=feature_names,
    background_data=X_train,
)

# SHAP values for the test set
shap_df = explainer.shap_values(X_test)

# LIME explanation for a single instance
lime_df = explainer.lime_explain(X_test.iloc[0], X_train, top_n=10)

# Permutation feature importance
perm_df = explainer.permutation_importance(X_test, y_test)

# Combined summary
summary = explainer.feature_importance_summary(X_test, y_test)
print(summary)
```

### Plotting

```python
from modelrisk.utils import RiskPlotter

plotter = RiskPlotter()
fig = plotter.roc_curve(y_true, y_score)
fig = plotter.cap_curve(y_true, y_score)
fig = plotter.reliability_diagram(y_true, y_score)
fig = plotter.loss_distribution(simulated_losses, var_level=0.99, cvar_level=0.975)
fig = plotter.shap_summary(shap_df)
fig.savefig("shap_summary.png", dpi=150)
```

---

### IFRS 9 — ScenarioManager (no-code entry point)

```python
from modelrisk.credit import ScenarioManager

# Option A — Python API (method chaining)
mgr = (
    ScenarioManager(discount_rate=0.05)
    .add_scenario("base",     weight=0.50, pd_scalar=1.0,  label="Central")
    .add_scenario("downside", weight=0.30, pd_scalar=1.80, label="Adverse")
    .add_scenario("upside",   weight=0.20, pd_scalar=0.70, label="Benign")
    .attach_portfolio(pd=df.pit_pd, lgd=df.lgd, ead=df.ead,
                      stage=df.stage, lifetime_pd=df.lifetime_pd)
)

# Option B — YAML config (no Python required for scenario changes)
mgr = ScenarioManager.from_yaml("scenarios.yaml")
mgr.attach_portfolio(pd=df.pit_pd, lgd=df.lgd, ead=df.ead, stage=df.stage)

results = mgr.run_all()
print(mgr.weighted_ecl())          # probability-weighted ECL
print(mgr.scenario_ecl_table())    # per-scenario ECL + weighted contribution
print(mgr.summary_report())        # audit-ready stage x scenario breakdown
mgr.to_yaml("scenarios_q4.yaml")   # export config for version control
```

### IRB — Through-the-cycle calibration and capital

```python
from modelrisk.credit.irb import TTCCalibrator, IRBCapital, IRBValidator

ttc = TTCCalibrator(min_pd=0.0003).fit(annual_default_rates)
irb = IRBCapital(asset_class="retail_mortgage")
rwa = irb.rwa_portfolio(pd_array=ttc_pds, lgd_array=lgds, ead_array=eads)
val = IRBValidator()
val.traffic_light_test(predicted_pd=0.008, observed_dr=0.015, n_obligors=500)
```

---

## Project Structure

```
modelrisk/
├── credit/
│   ├── base_pd.py           # BasePDModel abstract class
│   ├── pd.py                # LogisticPD, RandomForestPD, XGBoostPD, MertonPD
│   ├── lgd.py               # BetaLGD, LinearLGD
│   ├── scorecard.py         # Scorecard (WoE, IV, points)
│   ├── scenario_manager.py  # ScenarioManager — IFRS 9 orchestration
│   ├── ifrs9/
│   │   ├── pit_pd.py        # PITCalibrator (scalar, isotonic, Platt, time weights)
│   │   ├── staging.py       # StagingClassifier (dual SICR detection)
│   │   ├── forward_pd.py    # ForwardPDCurve (hazard-rate term structure)
│   │   ├── lifetime_pd.py   # LifetimePDCurve (discounted ECL per period)
│   │   ├── macro_overlay.py # MacroOverlay (OLS sensitivity + logit scenarios)
│   │   └── ecl.py           # ECLCalculator (portfolio ECL aggregation)
│   └── irb/
│       ├── ttc_pd.py        # TTCCalibrator (long-run average, Basel floor)
│       ├── smoothing.py     # CycleAdjuster (scalar, moving avg, HP filter)
│       ├── dr_mapping.py    # RatingMasterScale (log-spaced grade assignment)
│       ├── pit_to_ttc.py    # PITtoTTCBridge (scalar and logit-offset)
│       ├── capital.py       # IRBCapital (Basel ASRF RWA formula)
│       └── validation.py    # IRBValidator (traffic light + binomial backtest)
├── market/
│   ├── var.py          # HistoricalVaR, ParametricVaR, MonteCarloVaR
│   ├── cvar.py         # CVaR / Expected Shortfall
│   └── volatility.py   # EWMAVolatility, GARCHVolatility
├── operational/
│   ├── lda.py          # LossDistributionApproach
│   ├── scenarios.py    # ScenarioAnalysis, ExtremeValueModel
│   └── evt.py
├── evaluation/
│   ├── classification.py   # ClassificationMetrics
│   ├── regression.py       # RegressionMetrics
│   ├── calibration.py      # CalibrationMetrics
│   └── explainability.py   # Explainer (SHAP, LIME, permutation)
└── utils/
    ├── distributions.py    # DistributionFitter
    ├── simulation.py       # MonteCarloEngine
    └── plotting.py         # RiskPlotter
```

---

## Publishing a New Release

1. Bump the version in `pyproject.toml` and `modelrisk/__init__.py`
2. Commit and tag: `git tag v0.2.0 && git push origin v0.2.0`
3. The GitHub Actions `publish.yml` workflow builds and publishes to PyPI automatically via OIDC trusted publishing (no API key needed — configure once at pypi.org/manage/account/publishing/)

---

## Development

```bash
git clone https://github.com/saranshb1/modelRisk
cd modelRisk
pip install -e ".[all]"
pytest tests/ -v --cov=modelrisk
ruff check modelrisk/
```

---

## License

MIT — see [LICENSE](LICENSE).
