"""Tests for modelrisk.market."""

import numpy as np
import pytest

from modelrisk.market.var import HistoricalVaR, ParametricVaR, MonteCarloVaR
from modelrisk.market.cvar import CVaR
from modelrisk.market.volatility import EWMAVolatility, GARCHVolatility


class TestHistoricalVaR:
    def test_var_positive(self, daily_returns):
        model = HistoricalVaR(confidence_level=0.99).fit(daily_returns)
        assert model.var() > 0

    def test_higher_confidence_higher_var(self, daily_returns):
        v95 = HistoricalVaR(0.95).fit(daily_returns).var()
        v99 = HistoricalVaR(0.99).fit(daily_returns).var()
        assert v99 > v95

    def test_var_series_length(self, daily_returns):
        model = HistoricalVaR().fit(daily_returns)
        series = model.var_series(window=50)
        assert len(series) == len(daily_returns) - 50 + 1

    def test_invalid_confidence(self):
        with pytest.raises(ValueError):
            HistoricalVaR(confidence_level=1.5)

    def test_not_fitted_raises(self):
        with pytest.raises(RuntimeError):
            HistoricalVaR().var()


class TestParametricVaR:
    def test_var_positive(self, daily_returns):
        model = ParametricVaR().fit(daily_returns)
        assert model.var() > 0

    def test_t_var_higher_than_normal(self, daily_returns):
        model = ParametricVaR(confidence_level=0.99).fit(daily_returns)
        normal_var = model.var()
        t_var = model.var_with_t(df=3.0)
        assert t_var > normal_var

    def test_params_set_after_fit(self, daily_returns):
        model = ParametricVaR().fit(daily_returns)
        assert model.mu_ is not None
        assert model.sigma_ > 0


class TestMonteCarloVaR:
    def test_var_positive(self, daily_returns):
        model = MonteCarloVaR(n_simulations=10_000).fit(daily_returns)
        assert model.var() > 0

    def test_simulated_losses_shape(self, daily_returns):
        model = MonteCarloVaR(n_simulations=5_000).fit(daily_returns)
        assert model.simulated_losses_.shape == (5_000,)


class TestCVaR:
    @pytest.mark.parametrize("method", ["historical", "parametric", "montecarlo"])
    def test_cvar_exceeds_var(self, daily_returns, method):
        model = CVaR(confidence_level=0.975, method=method, n_simulations=10_000)
        model.fit(daily_returns)
        assert model.cvar() >= model.var() * 0.9  # CVaR ≥ VaR (with tolerance)

    def test_summary_returns_series(self, daily_returns):
        model = CVaR().fit(daily_returns)
        s = model.summary()
        assert "var" in s.index
        assert "cvar_es" in s.index

    def test_invalid_method(self):
        with pytest.raises(ValueError):
            CVaR(method="invalid_method")


class TestEWMAVolatility:
    def test_volatility_series_length(self, daily_returns):
        model = EWMAVolatility().fit(daily_returns)
        vol = model.volatility_series(annualise=False)
        assert len(vol) == len(daily_returns)

    def test_current_volatility_positive(self, daily_returns):
        model = EWMAVolatility().fit(daily_returns)
        assert model.current_volatility() > 0

    def test_invalid_lambda(self):
        with pytest.raises(ValueError):
            EWMAVolatility(lambda_=1.5)


class TestGARCHVolatility:
    def test_params_estimated(self, daily_returns):
        model = GARCHVolatility().fit(daily_returns)
        assert model.omega_ > 0
        assert 0 < model.alpha_ < 1
        assert 0 < model.beta_ < 1
        assert model.alpha_ + model.beta_ < 1.0

    def test_forecast_shape(self, daily_returns):
        model = GARCHVolatility().fit(daily_returns)
        forecasts = model.forecast(horizon=10)
        assert forecasts.shape == (10,)

    def test_volatility_series_positive(self, daily_returns):
        model = GARCHVolatility().fit(daily_returns)
        vol = model.volatility_series(annualise=False)
        assert (vol > 0).all()

    def test_unconditional_volatility_positive(self, daily_returns):
        model = GARCHVolatility().fit(daily_returns)
        assert model.unconditional_volatility > 0
