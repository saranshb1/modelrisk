"""Tests for modelrisk.operational."""

import numpy as np
import pytest

from modelrisk.operational.lda import LossDistributionApproach
from modelrisk.operational.scenarios import ScenarioAnalysis, ExtremeValueModel


class TestLDA:
    @pytest.fixture
    def lda_data(self):
        rng = np.random.default_rng(0)
        frequencies = rng.poisson(5, size=50)
        severities = rng.lognormal(10, 1.5, size=500)
        return frequencies, severities

    def test_fit_and_capital(self, lda_data):
        freq, sev = lda_data
        lda = LossDistributionApproach(n_simulations=10_000)
        lda.fit(freq, sev)
        capital = lda.capital_estimate()
        assert "var_capital" in capital
        assert "cvar_capital" in capital
        assert capital["cvar_capital"] >= capital["var_capital"]
        assert capital["expected_loss"] >= 0

    def test_invalid_distributions(self):
        with pytest.raises(ValueError):
            LossDistributionApproach(frequency_dist="invalid")
        with pytest.raises(ValueError):
            LossDistributionApproach(severity_dist="invalid")

    def test_simulate_shape(self, lda_data):
        freq, sev = lda_data
        lda = LossDistributionApproach(n_simulations=5_000)
        lda.fit(freq, sev)
        losses = lda.simulate()
        assert losses.shape == (5_000,)
        assert (losses >= 0).all()

    @pytest.mark.parametrize("freq_dist,sev_dist", [
        ("poisson", "lognormal"),
        ("poisson", "gamma"),
        ("negative_binomial", "lognormal"),
    ])
    def test_distribution_combinations(self, lda_data, freq_dist, sev_dist):
        freq, sev = lda_data
        lda = LossDistributionApproach(
            frequency_dist=freq_dist, severity_dist=sev_dist, n_simulations=5_000
        )
        lda.fit(freq, sev)
        capital = lda.capital_estimate()
        assert capital["var_capital"] > 0


class TestScenarioAnalysis:
    def test_add_scenario_and_eal(self):
        sa = ScenarioAnalysis(n_simulations=10_000)
        sa.add_scenario("cyber", frequency=0.5, severity_mean=100_000, severity_std=50_000)
        sa.add_scenario("fraud", frequency=2.0, severity_mean=20_000, severity_std=10_000)
        eal = sa.expected_annual_loss()
        assert len(eal) == 3  # 2 scenarios + TOTAL
        assert eal.iloc[-1]["scenario"] == "TOTAL"
        total_eal = eal.iloc[-1]["expected_annual_loss"]
        assert total_eal == pytest.approx(0.5 * 100_000 + 2.0 * 20_000)

    def test_simulate_shape(self):
        sa = ScenarioAnalysis(n_simulations=5_000)
        sa.add_scenario("it_failure", frequency=1.0, severity_mean=50_000, severity_std=20_000)
        losses = sa.simulate()
        assert losses.shape == (5_000,)


class TestEVT:
    def test_fit_and_var(self, loss_data):
        evt = ExtremeValueModel(threshold=None)
        evt.fit(loss_data)
        var = evt.var(0.999)
        assert var > 0

    def test_cvar_exceeds_var(self, loss_data):
        evt = ExtremeValueModel().fit(loss_data)
        assert evt.cvar(0.999) >= evt.var(0.999)

    def test_tail_summary(self, loss_data):
        evt = ExtremeValueModel().fit(loss_data)
        summary = evt.tail_summary()
        assert "xi_shape" in summary.index
        assert "beta_scale" in summary.index
        assert "var_99_9" in summary.index

    def test_not_fitted_raises(self):
        with pytest.raises(RuntimeError):
            ExtremeValueModel().var(0.999)

    def test_low_threshold_raises(self, loss_data):
        with pytest.raises(ValueError):
            ExtremeValueModel(threshold=1e9).fit(loss_data)
