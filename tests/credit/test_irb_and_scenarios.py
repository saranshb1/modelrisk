"""Tests for modelrisk.credit.irb and ScenarioManager."""

import numpy as np
import pandas as pd
import pytest

from modelrisk.credit.irb.ttc_pd import TTCCalibrator
from modelrisk.credit.irb.smoothing import CycleAdjuster, RatingMasterScale, IRBCapital, IRBValidator
from modelrisk.credit.irb.pit_to_ttc import PITtoTTCBridge
from modelrisk.credit.scenario_manager import ScenarioManager, Scenario


RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# TTCCalibrator
# ---------------------------------------------------------------------------

class TestTTCCalibrator:
    @pytest.fixture
    def historical_drs(self):
        return np.array([0.010, 0.018, 0.025, 0.035, 0.028, 0.020, 0.014])

    def test_long_run_average_correct(self, historical_drs):
        cal = TTCCalibrator().fit(historical_drs)
        assert abs(cal.long_run_average_pd - historical_drs.mean()) < 1e-9

    def test_floor_applied(self):
        cal = TTCCalibrator(min_pd=0.005).fit(np.array([0.001, 0.002, 0.001]))
        assert cal.long_run_average_pd >= 0.005

    def test_apply_scalar_preserves_ranking(self, historical_drs):
        cal = TTCCalibrator().fit(historical_drs)
        pit = np.array([0.01, 0.02, 0.05])
        ttc = cal.apply(pit)
        assert ttc[0] < ttc[1] < ttc[2]

    def test_apply_floored_at_min_pd(self, historical_drs):
        cal = TTCCalibrator(min_pd=0.0003).fit(historical_drs)
        ttc = cal.apply(np.array([0.00001, 0.00002]))
        assert ttc.min() >= 0.0003

    def test_weighted_fit(self, historical_drs):
        weights = np.arange(1, len(historical_drs) + 1, dtype=float)
        cal = TTCCalibrator().fit(historical_drs, weights=weights)
        expected = np.average(historical_drs, weights=weights)
        assert abs(cal.long_run_average_pd - expected) < 1e-9

    def test_calibration_summary_structure(self, historical_drs):
        cal = TTCCalibrator(cycle_length_years=5).fit(historical_drs)
        s = cal.calibration_summary()
        assert "long_run_avg_pd" in s.index
        assert "n_years_data" in s.index
        assert bool(s["data_covers_full_cycle"])  # 7 years >= 5

    def test_not_fitted_raises(self):
        with pytest.raises(RuntimeError):
            TTCCalibrator().long_run_average_pd


# ---------------------------------------------------------------------------
# CycleAdjuster
# ---------------------------------------------------------------------------

class TestCycleAdjuster:
    @pytest.fixture
    def pit_series(self):
        return np.array([0.010, 0.018, 0.030, 0.040, 0.035, 0.022, 0.015])

    @pytest.mark.parametrize("method", ["scalar", "moving_avg", "hp_filter"])
    def test_output_same_length(self, pit_series, method):
        adj = CycleAdjuster(method=method).smooth(pit_series)
        assert len(adj) == len(pit_series)

    def test_scalar_returns_constant(self, pit_series):
        adj = CycleAdjuster(method="scalar").smooth(pit_series)
        assert np.allclose(adj, adj[0])   # all values equal long-run mean

    def test_scalar_equals_mean(self, pit_series):
        adj = CycleAdjuster(method="scalar").smooth(pit_series)
        assert abs(adj[0] - pit_series.mean()) < 1e-9

    def test_moving_avg_smooths(self, pit_series):
        adj = CycleAdjuster(method="moving_avg", window=3).smooth(pit_series)
        # Variance of smooth should be less than variance of original
        assert np.var(adj) <= np.var(pit_series)

    def test_hp_filter_close_to_mean(self, pit_series):
        adj = CycleAdjuster(method="hp_filter", hp_lambda=1600).smooth(pit_series)
        # HP trend should be bounded between min and max of input
        assert adj.min() >= pit_series.min() * 0.5
        assert adj.max() <= pit_series.max() * 1.5

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            CycleAdjuster(method="fourier")


# ---------------------------------------------------------------------------
# RatingMasterScale
# ---------------------------------------------------------------------------

class TestRatingMasterScale:
    def test_default_scale_length(self):
        rms = RatingMasterScale(n_grades=18)
        assert len(rms.scale_table()) == 18

    def test_grade_assignment_range(self):
        rms = RatingMasterScale(n_grades=18)
        pds = np.array([0.0001, 0.001, 0.01, 0.05, 0.20, 0.80])
        grades = rms.assign_grades(pds)
        assert all(1 <= g <= 18 for g in grades)

    def test_higher_pd_higher_grade(self):
        rms = RatingMasterScale(n_grades=18)
        low_grade  = rms.assign_grades(np.array([0.001]))[0]
        high_grade = rms.assign_grades(np.array([0.20]))[0]
        assert high_grade > low_grade

    def test_grade_pd_retrieval(self):
        rms = RatingMasterScale(n_grades=18)
        pd_val = rms.grade_pd(1)
        assert pd_val > 0

    def test_invalid_grade_raises(self):
        rms = RatingMasterScale(n_grades=18)
        with pytest.raises(ValueError):
            rms.grade_pd(99)

    def test_scale_table_columns(self):
        rms = RatingMasterScale(n_grades=10)
        table = rms.scale_table()
        assert "grade" in table.columns
        assert "ttc_pd_pct" in table.columns

    def test_custom_scale(self):
        custom = pd.DataFrame({
            "grade": [1, 2, 3],
            "pd_lower": [0.0, 0.01, 0.05],
            "pd_upper": [0.01, 0.05, 1.0],
            "ttc_pd": [0.005, 0.03, 0.20],
        })
        rms = RatingMasterScale(n_grades=3, scale=custom)
        grades = rms.assign_grades(np.array([0.005, 0.03, 0.15]))
        assert grades[0] == 1
        assert grades[1] == 2
        assert grades[2] == 3


# ---------------------------------------------------------------------------
# PITtoTTCBridge
# ---------------------------------------------------------------------------

class TestPITtoTTCBridge:
    @pytest.fixture
    def paired_data(self):
        pit = RNG.uniform(0.01, 0.06, 200)
        ttc = pit * 0.65 + 0.005 + RNG.normal(0, 0.002, 200)
        ttc = np.clip(ttc, 0.001, 0.5)
        return pit, ttc

    @pytest.mark.parametrize("method", ["scalar", "logit_offset"])
    def test_convert_output_in_range(self, paired_data, method):
        pit, ttc = paired_data
        bridge = PITtoTTCBridge(method=method).fit(pit, ttc)
        converted = bridge.convert(pit)
        assert converted.min() >= 0.0003
        assert converted.max() <= 0.9999

    def test_scalar_preserves_ranking(self, paired_data):
        pit, ttc = paired_data
        bridge = PITtoTTCBridge(method="scalar").fit(pit, ttc)
        sorted_pit = np.sort(pit)
        converted = bridge.convert(sorted_pit)
        assert np.all(np.diff(converted) >= 0)

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            PITtoTTCBridge(method="regression")

    def test_not_fitted_raises(self):
        bridge = PITtoTTCBridge()
        with pytest.raises(RuntimeError):
            bridge.convert(np.array([0.01, 0.02]))


# ---------------------------------------------------------------------------
# IRBCapital
# ---------------------------------------------------------------------------

class TestIRBCapital:
    @pytest.mark.parametrize("asset_class", [
        "corporate", "retail_mortgage", "retail_other", "sme_retail"
    ])
    def test_rwa_positive_all_asset_classes(self, asset_class):
        irb = IRBCapital(asset_class=asset_class)
        result = irb.compute_rwa(pd=0.01, lgd=0.45, ead=1_000_000)
        assert result["rwa"] > 0

    def test_higher_pd_higher_rwa(self):
        irb = IRBCapital(asset_class="corporate")
        low  = irb.compute_rwa(pd=0.005, lgd=0.45, ead=1_000_000)["rwa"]
        high = irb.compute_rwa(pd=0.050, lgd=0.45, ead=1_000_000)["rwa"]
        assert high > low

    def test_higher_lgd_higher_rwa(self):
        irb = IRBCapital(asset_class="corporate")
        low  = irb.compute_rwa(pd=0.01, lgd=0.25, ead=1_000_000)["rwa"]
        high = irb.compute_rwa(pd=0.01, lgd=0.65, ead=1_000_000)["rwa"]
        assert high > low

    def test_min_pd_floor_applied(self):
        irb = IRBCapital(asset_class="corporate")
        result = irb.compute_rwa(pd=0.00001, lgd=0.45, ead=1_000_000)
        assert result["pd"] >= 0.0003

    def test_portfolio_rwa_total_row(self):
        irb = IRBCapital(asset_class="corporate")
        portfolio = irb.rwa_portfolio(
            pd_array=[0.005, 0.02, 0.10],
            lgd_array=[0.45, 0.45, 0.45],
            ead_array=[1_000_000, 500_000, 200_000],
        )
        assert "TOTAL" in portfolio.index
        assert portfolio.loc["TOTAL", "rwa"] > 0

    def test_portfolio_total_rwa_sums(self):
        irb = IRBCapital(asset_class="retail_mortgage")
        pds  = [0.003, 0.008, 0.015]
        lgds = [0.15, 0.15, 0.15]
        eads = [200_000, 300_000, 150_000]
        portfolio = irb.rwa_portfolio(pds, lgds, eads)
        individual_sum = portfolio.drop("TOTAL")["rwa"].sum()
        total = float(portfolio.loc["TOTAL", "rwa"])
        assert abs(individual_sum - total) < 1.0

    def test_invalid_asset_class_raises(self):
        with pytest.raises(ValueError):
            IRBCapital(asset_class="unknown")


# ---------------------------------------------------------------------------
# IRBValidator
# ---------------------------------------------------------------------------

class TestIRBValidator:
    def test_traffic_light_green(self):
        val = IRBValidator()
        result = val.traffic_light_test(
            predicted_pd=0.010, observed_dr=0.011, n_obligors=1000
        )
        assert result["zone"] == "green"

    def test_traffic_light_red(self):
        val = IRBValidator()
        result = val.traffic_light_test(
            predicted_pd=0.010, observed_dr=0.025, n_obligors=1000
        )
        assert result["zone"] in ("amber", "red")

    def test_traffic_light_structure(self):
        val = IRBValidator()
        result = val.traffic_light_test(0.01, 0.015, 500)
        assert set(result.keys()).issuperset(
            {"zone", "relative_deviation", "predicted_pd", "observed_dr"}
        )

    def test_binomial_test_p_value_range(self):
        val = IRBValidator()
        result = val.binomial_test(0.008, 0.015, 500)
        assert 0 <= result["p_value"] <= 1

    def test_binomial_reject_h0_when_severe(self):
        val = IRBValidator()
        # Observed 10× predicted — should strongly reject
        result = val.binomial_test(0.005, 0.05, 2000)
        assert result["reject_h0"] is True

    def test_binomial_dont_reject_when_aligned(self):
        val = IRBValidator()
        result = val.binomial_test(0.010, 0.011, 500)
        assert result["reject_h0"] is False

    def test_portfolio_backtest_shape(self):
        val = IRBValidator()
        df = pd.DataFrame({
            "predicted_pd": [0.005, 0.015, 0.04, 0.10],
            "observed_dr":  [0.006, 0.018, 0.035, 0.12],
            "n_obligors":   [1000, 500, 200, 100],
        })
        result = val.portfolio_backtest(df)
        assert len(result) == 4
        assert "zone" in result.columns
        assert "p_value" in result.columns

    def test_invalid_pd_raises(self):
        val = IRBValidator()
        with pytest.raises(ValueError):
            val.traffic_light_test(predicted_pd=0, observed_dr=0.01, n_obligors=100)


# ---------------------------------------------------------------------------
# Scenario (dataclass)
# ---------------------------------------------------------------------------

class TestScenarioDataclass:
    def test_valid_scenario(self):
        s = Scenario(name="base", weight=0.5, pd_scalar=1.0, label="Central")
        assert s.label == "Central"

    def test_label_defaults_to_name(self):
        s = Scenario(name="downside", weight=0.3)
        assert s.label == "Downside"

    def test_invalid_weight_raises(self):
        with pytest.raises(ValueError):
            Scenario(name="bad", weight=1.5)

    def test_invalid_pd_scalar_raises(self):
        with pytest.raises(ValueError):
            Scenario(name="bad", weight=0.5, pd_scalar=-1.0)


# ---------------------------------------------------------------------------
# ScenarioManager
# ---------------------------------------------------------------------------

class TestScenarioManager:
    @pytest.fixture
    def portfolio(self):
        n = 400
        return {
            "pd":    RNG.uniform(0.005, 0.08, n),
            "lgd":   RNG.uniform(0.20, 0.55, n),
            "ead":   RNG.uniform(10_000, 500_000, n),
            "stage": RNG.choice([1, 2, 3], n, p=[0.70, 0.25, 0.05]),
        }

    @pytest.fixture
    def configured_manager(self, portfolio):
        mgr = (
            ScenarioManager(discount_rate=0.05)
            .add_scenario("base",     weight=0.50, pd_scalar=1.0,  label="Central")
            .add_scenario("downside", weight=0.30, pd_scalar=1.80, label="Adverse")
            .add_scenario("upside",   weight=0.20, pd_scalar=0.70, label="Benign")
        )
        mgr.attach_portfolio(**portfolio)
        mgr.run_all()
        return mgr

    # --- Configuration ---

    def test_add_scenario_method_chaining(self, portfolio):
        mgr = (
            ScenarioManager()
            .add_scenario("a", weight=0.5)
            .add_scenario("b", weight=0.5)
        )
        assert len(mgr._scenarios) == 2

    def test_list_scenarios_structure(self, configured_manager):
        df = configured_manager.list_scenarios()
        assert len(df) == 3
        assert "weight" in df.columns
        assert "pd_scalar" in df.columns

    def test_remove_scenario(self, portfolio):
        mgr = ScenarioManager(validate_weights=False)
        mgr.add_scenario("a", weight=0.5).add_scenario("b", weight=0.5)
        mgr.remove_scenario("a")
        assert "a" not in mgr._scenarios

    def test_remove_nonexistent_raises(self, portfolio):
        mgr = ScenarioManager(validate_weights=False)
        mgr.add_scenario("a", weight=1.0)
        with pytest.raises(KeyError):
            mgr.remove_scenario("nonexistent")

    def test_replace_scenario(self, portfolio):
        mgr = ScenarioManager(validate_weights=False)
        mgr.add_scenario("base", weight=0.5, pd_scalar=1.0)
        mgr.add_scenario("base", weight=0.6, pd_scalar=1.2)  # replace
        assert len(mgr._scenarios) == 1
        assert mgr._scenarios["base"].pd_scalar == 1.2

    # --- Portfolio attachment ---

    def test_attach_portfolio_sets_data(self, portfolio):
        mgr = ScenarioManager(validate_weights=False)
        mgr.attach_portfolio(**portfolio)
        assert len(mgr._portfolio["ead"]) == 400

    def test_attach_without_stage_defaults_to_1(self, portfolio):
        mgr = ScenarioManager(validate_weights=False)
        p = {k: v for k, v in portfolio.items() if k != "stage"}
        mgr.attach_portfolio(**p)
        assert all(mgr._portfolio["stage"] == 1)

    def test_attach_without_lifetime_pd_approximates(self, portfolio):
        mgr = ScenarioManager(validate_weights=False)
        p = {k: v for k, v in portfolio.items()}
        mgr.attach_portfolio(**p)
        assert mgr._portfolio["lifetime_pd"] is not None

    # --- Execution ---

    def test_run_scenario_returns_dataframe(self, portfolio):
        mgr = ScenarioManager(validate_weights=False)
        mgr.add_scenario("base", weight=1.0, pd_scalar=1.0)
        mgr.attach_portfolio(**portfolio)
        result = mgr.run_scenario("base")
        assert isinstance(result, pd.DataFrame)
        assert "ecl" in result.columns

    def test_run_nonexistent_scenario_raises(self, portfolio):
        mgr = ScenarioManager(validate_weights=False)
        mgr.add_scenario("base", weight=1.0)
        mgr.attach_portfolio(**portfolio)
        with pytest.raises(KeyError):
            mgr.run_scenario("nonexistent")

    def test_run_all_returns_all_scenarios(self, configured_manager):
        assert set(configured_manager._results.keys()) == {"base", "downside", "upside"}

    def test_weight_validation_error(self, portfolio):
        mgr = ScenarioManager(validate_weights=True)
        mgr.add_scenario("a", weight=0.6)
        mgr.add_scenario("b", weight=0.6)   # sum = 1.2
        mgr.attach_portfolio(**portfolio)
        with pytest.raises(ValueError, match="sum to 1.0"):
            mgr.run_all()

    def test_no_portfolio_raises(self):
        mgr = ScenarioManager(validate_weights=False)
        mgr.add_scenario("base", weight=1.0)
        with pytest.raises(RuntimeError, match="portfolio"):
            mgr.run_all()

    def test_no_scenarios_raises(self, portfolio):
        mgr = ScenarioManager(validate_weights=False)
        mgr.attach_portfolio(**portfolio)
        with pytest.raises(RuntimeError, match="scenarios"):
            mgr.run_all()

    # --- Results ---

    def test_weighted_ecl_positive(self, configured_manager):
        assert configured_manager.weighted_ecl() > 0

    def test_downside_ecl_exceeds_base(self, configured_manager):
        assert (configured_manager._ecl_totals["downside"]
                > configured_manager._ecl_totals["base"])

    def test_upside_ecl_below_base(self, configured_manager):
        assert (configured_manager._ecl_totals["upside"]
                < configured_manager._ecl_totals["base"])

    def test_weighted_ecl_between_min_and_max(self, configured_manager):
        min_ecl = min(configured_manager._ecl_totals.values())
        max_ecl = max(configured_manager._ecl_totals.values())
        assert min_ecl <= configured_manager.weighted_ecl() <= max_ecl

    def test_scenario_ecl_table_has_total_row(self, configured_manager):
        tbl = configured_manager.scenario_ecl_table()
        assert "WEIGHTED TOTAL" in tbl["scenario"].values

    def test_scenario_ecl_table_weights_sum_to_1(self, configured_manager):
        tbl = configured_manager.scenario_ecl_table()
        data_rows = tbl[tbl["scenario"] != "WEIGHTED TOTAL"]
        assert abs(data_rows["weight"].sum() - 1.0) < 1e-9

    def test_summary_report_has_total_row(self, configured_manager):
        report = configured_manager.summary_report()
        assert "WEIGHTED TOTAL" in report["scenario"].values

    def test_summary_report_has_required_columns(self, configured_manager):
        report = configured_manager.summary_report()
        required = {"scenario", "label", "weight", "stage", "weighted_ecl"}
        assert required.issubset(set(report.columns))

    def test_exposure_level_all_scenarios(self, configured_manager, portfolio):
        all_exp = configured_manager.exposure_level_results()
        assert len(all_exp) == 400 * 3

    def test_exposure_level_single_scenario(self, configured_manager, portfolio):
        one = configured_manager.exposure_level_results("base")
        assert len(one) == 400
        assert (one["scenario"] == "base").all()

    def test_exposure_level_invalid_scenario_raises(self, configured_manager):
        with pytest.raises(KeyError):
            configured_manager.exposure_level_results("nonexistent")

    def test_weighted_ecl_before_run_raises(self, portfolio):
        mgr = ScenarioManager(validate_weights=False)
        mgr.add_scenario("base", weight=1.0)
        mgr.attach_portfolio(**portfolio)
        with pytest.raises(RuntimeError):
            mgr.weighted_ecl()

    # --- Serialisation ---

    def test_from_dict_creates_correct_scenarios(self):
        config = {
            "discount_rate": 0.04,
            "scenarios": {
                "base":  {"weight": 0.60, "pd_scalar": 1.0,  "label": "Base"},
                "stress":{"weight": 0.40, "pd_scalar": 2.0,  "label": "Stress"},
            }
        }
        mgr = ScenarioManager.from_dict(config)
        assert len(mgr._scenarios) == 2
        assert mgr.discount_rate == 0.04
        assert mgr._scenarios["stress"].pd_scalar == 2.0

    def test_from_dict_weights_preserved(self):
        config = {
            "scenarios": {
                "a": {"weight": 0.7, "pd_scalar": 1.0},
                "b": {"weight": 0.3, "pd_scalar": 1.5},
            }
        }
        mgr = ScenarioManager.from_dict(config)
        assert abs(mgr._scenarios["a"].weight - 0.7) < 1e-9

    def test_repr_contains_key_info(self, configured_manager):
        r = repr(configured_manager)
        assert "ScenarioManager" in r
        assert "3" in r    # 3 scenarios
