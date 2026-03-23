"""Tests for modelrisk.credit.ifrs9."""

import numpy as np
import pandas as pd
import pytest

from modelrisk.credit.ifrs9.pit_pd import PITCalibrator
from modelrisk.credit.ifrs9.staging import StagingClassifier
from modelrisk.credit.ifrs9.forward_pd import ForwardPDCurve
from modelrisk.credit.ifrs9.lifetime_pd import LifetimePDCurve
from modelrisk.credit.ifrs9.macro_overlay import MacroOverlay
from modelrisk.credit.ifrs9.ecl import ECLCalculator


RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# PITCalibrator
# ---------------------------------------------------------------------------

class TestPITCalibrator:
    @pytest.fixture
    def raw_pd(self):
        return RNG.uniform(0.005, 0.08, 200)

    @pytest.fixture
    def binary_outcomes(self, raw_pd):
        return (RNG.random(200) < raw_pd).astype(int)

    def test_scalar_calibration_scales_correctly(self, raw_pd):
        cal = PITCalibrator(method="scalar")
        cal.calibrate(raw_pd, observed_default_rate=0.035, model_long_run_dr=0.020)
        pit = cal.transform(raw_pd)
        expected_scalar = 0.035 / 0.020
        assert abs(pit.mean() / raw_pd.mean() - expected_scalar) < 0.01

    def test_scalar_output_in_range(self, raw_pd):
        cal = PITCalibrator(method="scalar", min_pd=0.0001, max_pd=0.9999)
        cal.calibrate(raw_pd, observed_default_rate=0.04, model_long_run_dr=0.02)
        pit = cal.transform(raw_pd)
        assert pit.min() >= 0.0001
        assert pit.max() <= 0.9999

    def test_isotonic_monotone(self, raw_pd, binary_outcomes):
        sorted_pd = np.sort(raw_pd)
        sorted_y = binary_outcomes[np.argsort(raw_pd)]
        cal = PITCalibrator(method="isotonic")
        cal.calibrate(sorted_pd, y_cal=sorted_y)
        pit = cal.transform(sorted_pd)
        # Isotonic output must be non-decreasing
        assert np.all(np.diff(pit) >= -1e-9)

    def test_platt_output_in_range(self, raw_pd, binary_outcomes):
        cal = PITCalibrator(method="platt")
        cal.calibrate(raw_pd, y_cal=binary_outcomes)
        pit = cal.transform(raw_pd)
        assert pit.min() >= 0
        assert pit.max() <= 1

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            PITCalibrator(method="unknown")

    def test_scalar_missing_args_raises(self, raw_pd):
        cal = PITCalibrator(method="scalar")
        with pytest.raises(ValueError):
            cal.calibrate(raw_pd)

    def test_isotonic_missing_y_raises(self, raw_pd):
        cal = PITCalibrator(method="isotonic")
        with pytest.raises(ValueError):
            cal.calibrate(raw_pd)

    def test_not_fitted_raises(self):
        cal = PITCalibrator(method="scalar")
        with pytest.raises(RuntimeError):
            cal.transform(np.array([0.01, 0.02]))

    def test_exponential_weights_recency(self):
        dates = pd.date_range("2020-01-01", periods=36, freq="ME")
        w = PITCalibrator.exponential_weights(
            pd.Series(dates), halflife_months=18
        )
        assert len(w) == 36
        # Recent observations (last) should have higher weight
        assert w.iloc[-1] > w.iloc[0]
        # Weights normalised to mean ~1.0
        assert abs(w.mean() - 1.0) < 0.05

    def test_exponential_weights_halflife(self):
        dates = pd.date_range("2020-01-01", periods=24, freq="ME")
        w = PITCalibrator.exponential_weights(pd.Series(dates), halflife_months=12)
        # Observation 12 months before reference should have weight ~0.5
        assert abs(w.iloc[-13] / w.iloc[-1] - 0.5) < 0.05

    def test_calibration_summary_scalar(self, raw_pd):
        cal = PITCalibrator(method="scalar")
        cal.calibrate(raw_pd, observed_default_rate=0.03, model_long_run_dr=0.015)
        s = cal.calibration_summary()
        assert "scalar_multiplier" in s.index
        assert abs(s["scalar_multiplier"] - 2.0) < 0.01


# ---------------------------------------------------------------------------
# StagingClassifier
# ---------------------------------------------------------------------------

class TestStagingClassifier:
    @pytest.fixture
    def sample_data(self):
        cur  = np.array([0.001, 0.002, 0.015, 0.04, 0.25, 0.35])
        orig = np.array([0.001, 0.001, 0.005, 0.015, 0.10, 0.10])
        return cur, orig

    def test_stage3_above_default_threshold(self, sample_data):
        cur, orig = sample_data
        clf = StagingClassifier(method="dual", default_threshold=0.20)
        stages = clf.classify(cur, orig)
        assert stages[4] == 3
        assert stages[5] == 3

    def test_stage2_absolute_trigger(self, sample_data):
        cur, orig = sample_data
        clf = StagingClassifier(method="absolute", absolute_threshold=0.01)
        stages = clf.classify(cur, orig)
        # cur[2] = 0.015 > threshold 0.01
        assert stages[2] == 2

    def test_stage2_relative_trigger(self, sample_data):
        cur, orig = sample_data
        clf = StagingClassifier(method="relative", relative_multiplier=2.0)
        stages = clf.classify(cur, orig)
        # cur[1]=0.002, orig[1]=0.001 → ratio=2.0 ≥ 2.0 → Stage 2
        assert stages[1] == 2

    def test_dual_fires_on_either(self, sample_data):
        cur, orig = sample_data
        clf = StagingClassifier(
            method="dual",
            absolute_threshold=0.03,
            relative_multiplier=10.0,  # very high — relative won't fire
        )
        stages = clf.classify(cur, orig)
        # cur[3]=0.04 > 0.03 → absolute fires → Stage 2
        assert stages[3] == 2

    def test_stage1_low_credit_risk_exemption(self):
        cur = np.array([0.002, 0.002])
        orig = np.array([0.001, 0.001])  # relative = 2x, would trigger
        clf = StagingClassifier(
            method="relative",
            relative_multiplier=1.5,
            low_credit_risk_exemption=0.003,
        )
        stages = clf.classify(cur, orig)
        # cur < 0.003 → exemption → Stage 1 despite relative trigger
        assert all(stages == 1)

    def test_is_defaulted_overrides_to_stage3(self):
        cur = np.array([0.001, 0.001])
        is_def = np.array([False, True])
        clf = StagingClassifier(method="absolute")
        stages = clf.classify(cur, is_defaulted=is_def)
        assert stages[0] == 1
        assert stages[1] == 3

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            StagingClassifier(method="invalid")

    def test_relative_missing_origination_raises(self):
        clf = StagingClassifier(method="relative")
        with pytest.raises(ValueError):
            clf.classify(np.array([0.01, 0.02]))

    def test_stage_summary_structure(self, sample_data):
        cur, orig = sample_data
        clf = StagingClassifier(method="dual")
        stages = clf.classify(cur, orig)
        summary = clf.stage_summary(stages, current_pd=cur, exposure_at_default=cur * 1e6)
        assert len(summary) == 3
        assert set(summary.columns).issuperset({"stage", "count", "pct_count", "mean_pd"})
        assert abs(summary["pct_count"].sum() - 100.0) < 0.01

    def test_stage_counts_sum_to_total(self, sample_data):
        cur, orig = sample_data
        clf = StagingClassifier(method="dual")
        stages = clf.classify(cur, orig)
        summary = clf.stage_summary(stages)
        assert summary["count"].sum() == len(cur)


# ---------------------------------------------------------------------------
# ForwardPDCurve
# ---------------------------------------------------------------------------

class TestForwardPDCurve:
    def test_constant_hazard_shape(self):
        curve = ForwardPDCurve(n_periods=60)
        marginal = curve.build(pit_pd_12m=0.03)
        assert marginal.shape == (60,)

    def test_constant_hazard_non_negative(self):
        curve = ForwardPDCurve(n_periods=60)
        marginal = curve.build(pit_pd_12m=0.03)
        assert marginal.min() >= 0
        assert marginal.max() <= 1

    def test_cumulative_pd_increases(self):
        curve = ForwardPDCurve(n_periods=60)
        curve.build(pit_pd_12m=0.03)
        cum = curve.cumulative_pd()
        assert np.all(np.diff(cum) >= -1e-9)

    def test_higher_pd_higher_cumulative(self):
        low = ForwardPDCurve(n_periods=24)
        low.build(pit_pd_12m=0.01)
        high = ForwardPDCurve(n_periods=24)
        high.build(pit_pd_12m=0.08)
        assert high.cumulative_pd()[-1] > low.cumulative_pd()[-1]

    def test_seasoning_curve_custom_factors(self):
        curve = ForwardPDCurve(n_periods=12)
        factors = np.ones(12)
        factors[0] = 0.5   # lower default in first period
        marginal = curve.build(pit_pd_12m=0.03, method="seasoning_curve",
                               seasoning_factors=factors)
        assert marginal.shape == (12,)
        assert marginal[0] < marginal[5]   # first period lower due to factor

    def test_seasoning_wrong_length_raises(self):
        curve = ForwardPDCurve(n_periods=12)
        with pytest.raises(ValueError):
            curve.build(pit_pd_12m=0.03, method="seasoning_curve",
                        seasoning_factors=np.ones(10))

    def test_invalid_method_raises(self):
        curve = ForwardPDCurve(n_periods=12)
        with pytest.raises(ValueError):
            curve.build(pit_pd_12m=0.03, method="unknown_method")

    def test_as_dataframe_columns(self):
        curve = ForwardPDCurve(n_periods=12)
        curve.build(pit_pd_12m=0.03)
        df = curve.as_dataframe()
        assert set(df.columns) == {"period", "marginal_pd", "cumulative_pd", "survival"}
        assert len(df) == 12

    def test_not_fitted_raises(self):
        curve = ForwardPDCurve(n_periods=12)
        with pytest.raises(RuntimeError):
            curve.cumulative_pd()


# ---------------------------------------------------------------------------
# LifetimePDCurve
# ---------------------------------------------------------------------------

class TestLifetimePDCurve:
    @pytest.fixture
    def marginal_pds(self):
        fwd = ForwardPDCurve(n_periods=60)
        return fwd.build(pit_pd_12m=0.03)

    def test_lifetime_pd_in_range(self, marginal_pds):
        lc = LifetimePDCurve().compute(marginal_pds, remaining_periods=48)
        assert 0 < lc.lifetime_pd < 1

    def test_longer_remaining_higher_lifetime_pd(self, marginal_pds):
        lc_short = LifetimePDCurve().compute(marginal_pds, 12)
        lc_long  = LifetimePDCurve().compute(marginal_pds, 48)
        assert lc_long.lifetime_pd > lc_short.lifetime_pd

    def test_ecl_weights_shape(self, marginal_pds):
        lc = LifetimePDCurve(discount_rate=0.05).compute(marginal_pds, 36)
        df = lc.discounted_ecl_weights(lgd=0.45, ead=100_000)
        assert len(df) == 36
        assert "ecl_contribution" in df.columns
        assert "cumulative_ecl" in df.columns

    def test_ecl_contributions_non_negative(self, marginal_pds):
        lc = LifetimePDCurve().compute(marginal_pds, 36)
        df = lc.discounted_ecl_weights(lgd=0.45, ead=100_000)
        assert (df["ecl_contribution"] >= 0).all()

    def test_total_ecl_positive(self, marginal_pds):
        lc = LifetimePDCurve(discount_rate=0.05).compute(marginal_pds, 48)
        assert lc.total_ecl(lgd=0.45, ead=100_000) > 0

    def test_higher_lgd_higher_ecl(self, marginal_pds):
        lc = LifetimePDCurve().compute(marginal_pds, 36)
        ecl_low  = lc.total_ecl(lgd=0.20, ead=100_000)
        ecl_high = lc.total_ecl(lgd=0.60, ead=100_000)
        assert ecl_high > ecl_low

    def test_remaining_exceeds_marginal_raises(self, marginal_pds):
        with pytest.raises(ValueError):
            LifetimePDCurve().compute(marginal_pds, remaining_periods=999)

    def test_not_fitted_raises(self):
        with pytest.raises(RuntimeError):
            LifetimePDCurve().lifetime_pd

    def test_discount_factors_decreasing(self, marginal_pds):
        lc = LifetimePDCurve(discount_rate=0.05).compute(marginal_pds, 24)
        df = lc.discount_factors()
        assert np.all(np.diff(df) < 0)


# ---------------------------------------------------------------------------
# MacroOverlay
# ---------------------------------------------------------------------------

class TestMacroOverlay:
    @pytest.fixture
    def macro_data(self):
        n = 40
        t = np.linspace(0, 4 * np.pi, n)
        gdp = np.sin(t) * 2 + 1.5
        unemp = -np.sin(t) * 1.5 + 6
        hist_pd = 0.015 + 0.005 * np.sin(t + 0.3)
        return pd.DataFrame({"gdp": gdp, "unemployment": unemp}), hist_pd

    def test_fit_sensitivity_sets_coef(self, macro_data):
        macro_df, hist_pd = macro_data
        ov = MacroOverlay(method="logit_link")
        ov.fit_sensitivity(hist_pd, macro_df)
        assert ov._coef is not None
        assert len(ov._coef) == 2

    def test_apply_base_returns_unchanged(self, macro_data):
        macro_df, hist_pd = macro_data
        ov = MacroOverlay().fit_sensitivity(hist_pd, macro_df)
        baseline = {"gdp": 1.5, "unemployment": 6.0}
        result = ov.apply(0.025, baseline, baseline)
        assert abs(result - 0.025) < 0.005   # same macro → near-unchanged

    def test_apply_downside_increases_pd(self, macro_data):
        macro_df, hist_pd = macro_data
        ov = MacroOverlay(method="logit_link").fit_sensitivity(hist_pd, macro_df)
        baseline  = {"gdp": 1.5, "unemployment": 6.0}
        downside  = {"gdp": -1.0, "unemployment": 9.0}
        adjusted  = ov.apply(0.025, downside, baseline)
        assert adjusted > 0.025   # worse macro → higher PD

    def test_apply_upside_decreases_pd(self, macro_data):
        macro_df, hist_pd = macro_data
        ov = MacroOverlay(method="logit_link").fit_sensitivity(hist_pd, macro_df)
        baseline = {"gdp": 1.5, "unemployment": 6.0}
        upside   = {"gdp": 3.5, "unemployment": 4.0}
        adjusted = ov.apply(0.025, upside, baseline)
        assert adjusted < 0.025

    def test_pd_stays_in_valid_range(self, macro_data):
        macro_df, hist_pd = macro_data
        ov = MacroOverlay().fit_sensitivity(hist_pd, macro_df)
        baseline = {"gdp": 1.5, "unemployment": 6.0}
        extreme  = {"gdp": -10.0, "unemployment": 25.0}
        result = ov.apply(0.5, extreme, baseline)
        assert 0 < result < 1

    def test_sensitivity_summary_structure(self, macro_data):
        macro_df, hist_pd = macro_data
        ov = MacroOverlay().fit_sensitivity(hist_pd, macro_df)
        s = ov.sensitivity_summary()
        assert "variable" in s.columns
        assert "coefficient" in s.columns
        assert len(s) == 2

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            MacroOverlay(method="bad")

    def test_not_fitted_raises(self):
        ov = MacroOverlay()
        with pytest.raises(RuntimeError):
            ov.apply(0.025, {}, {})


# ---------------------------------------------------------------------------
# ECLCalculator
# ---------------------------------------------------------------------------

class TestECLCalculator:
    @pytest.fixture
    def portfolio(self):
        n = 300
        return {
            "pd":          RNG.uniform(0.005, 0.08, n),
            "lgd":         RNG.uniform(0.20, 0.60, n),
            "ead":         RNG.uniform(10_000, 500_000, n),
            "stage":       RNG.choice([1, 2, 3], n, p=[0.70, 0.25, 0.05]),
            "lifetime_pd": RNG.uniform(0.05, 0.40, n),
        }

    def test_compute_portfolio_shape(self, portfolio):
        calc = ECLCalculator(discount_rate=0.05)
        result = calc.compute_portfolio(**portfolio)
        assert len(result) == 300

    def test_ecl_non_negative(self, portfolio):
        calc = ECLCalculator(discount_rate=0.05)
        result = calc.compute_portfolio(**portfolio)
        assert (result["ecl"] >= 0).all()

    def test_stage1_uses_12m_pd(self, portfolio):
        calc = ECLCalculator()
        result = calc.compute_portfolio(**portfolio)
        stage1 = result[result["stage"] == 1]
        assert (stage1["pd_used"] == stage1["pd_12m"]).all()

    def test_stage2_uses_lifetime_pd(self, portfolio):
        calc = ECLCalculator()
        result = calc.compute_portfolio(**portfolio)
        stage2 = result[result["stage"] == 2]
        if len(stage2) > 0:
            assert (stage2["pd_used"] == stage2["lifetime_pd"]).all()

    def test_summary_has_total_row(self, portfolio):
        calc = ECLCalculator()
        result = calc.compute_portfolio(**portfolio)
        summary = calc.summary(result)
        assert "TOTAL" in summary["stage"].astype(str).values

    def test_summary_ecl_sums_correctly(self, portfolio):
        calc = ECLCalculator()
        result = calc.compute_portfolio(**portfolio)
        summary = calc.summary(result)
        total_ecl = float(summary[summary["stage"].astype(str) == "TOTAL"]["total_ecl"].iloc[0])
        assert abs(total_ecl - result["ecl"].sum()) < 0.01

    def test_weighted_ecl_calculation(self):
        calc = ECLCalculator()
        w_ecl = calc.weighted_ecl(
            scenario_ecls={"base": 1_000_000, "down": 1_800_000, "up": 600_000},
            scenario_weights={"base": 0.50, "down": 0.30, "up": 0.20},
        )
        expected = 0.5 * 1_000_000 + 0.3 * 1_800_000 + 0.2 * 600_000
        assert abs(w_ecl - expected) < 1.0

    def test_weighted_ecl_bad_weights_raises(self):
        calc = ECLCalculator()
        with pytest.raises(ValueError):
            calc.weighted_ecl(
                scenario_ecls={"a": 100, "b": 200},
                scenario_weights={"a": 0.6, "b": 0.6},   # sum = 1.2
            )

    def test_higher_pd_scalar_higher_ecl(self):
        calc = ECLCalculator()
        pd_arr  = np.full(100, 0.02)
        lgd_arr = np.full(100, 0.45)
        ead_arr = np.full(100, 100_000)
        stage   = np.ones(100, dtype=int)
        base = calc.compute_portfolio(pd_arr, lgd_arr, ead_arr, stage)["ecl"].sum()
        stress = calc.compute_portfolio(
            pd_arr * 2, lgd_arr, ead_arr, stage
        )["ecl"].sum()
        assert stress > base
