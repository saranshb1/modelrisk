"""Tests for modelrisk.credit."""

import numpy as np
import pandas as pd
import pytest

from modelrisk.credit.pd import LogisticPD, MertonPD
from modelrisk.credit.lgd import BetaLGD, LinearLGD
from modelrisk.credit.scorecard import Scorecard


class TestLogisticPD:
    def test_fit_predict(self, binary_classification_data):
        X_train, X_test, y_train, y_test = binary_classification_data
        model = LogisticPD()
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)
        assert proba.shape == (len(X_test),)
        assert np.all((proba >= 0) & (proba <= 1))

    def test_coefficient_summary(self, binary_classification_data):
        X_train, _, y_train, _ = binary_classification_data
        model = LogisticPD().fit(X_train, y_train)
        summary = model.coefficient_summary()
        assert "feature" in summary.columns
        assert "coefficient" in summary.columns
        assert "odds_ratio" in summary.columns
        assert len(summary) == X_train.shape[1]

    def test_numpy_input(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(200, 5))
        y = (rng.random(200) > 0.7).astype(int)
        model = LogisticPD(scale_features=False)
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (200,)

    def test_not_fitted_raises(self):
        model = LogisticPD()
        with pytest.raises(Exception):
            model.predict_proba(np.zeros((5, 3)))


class TestMertonPD:
    def test_estimate_pd_returns_dict(self):
        model = MertonPD(risk_free_rate=0.05, time_horizon=1.0)
        result = model.estimate_pd(
            equity_value=50.0,
            equity_volatility=0.30,
            debt_face_value=80.0,
        )
        assert "pd" in result
        assert "distance_to_default" in result
        assert "asset_value" in result
        assert 0 <= result["pd"] <= 1

    def test_higher_debt_increases_pd(self):
        model = MertonPD()
        low_debt = model.estimate_pd(50.0, 0.30, 40.0)["pd"]
        high_debt = model.estimate_pd(50.0, 0.30, 120.0)["pd"]
        assert high_debt > low_debt

    def test_higher_volatility_increases_pd(self):
        model = MertonPD()
        low_vol = model.estimate_pd(50.0, 0.10, 80.0)["pd"]
        high_vol = model.estimate_pd(50.0, 0.60, 80.0)["pd"]
        assert high_vol > low_vol


class TestLGD:
    @pytest.fixture
    def lgd_data(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(300, 4))
        y = np.clip(rng.beta(2, 5, size=300), 0.01, 0.99)
        return pd.DataFrame(X, columns=list("abcd")), pd.Series(y)

    def test_linear_lgd(self, lgd_data):
        X, y = lgd_data
        model = LinearLGD()
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (len(X),)
        assert np.all((preds >= 0) & (preds <= 1))

    def test_beta_lgd(self, lgd_data):
        X, y = lgd_data
        model = BetaLGD()
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (len(X),)
        assert np.all((preds >= 0) & (preds <= 1))
        assert model.phi_ is not None and model.phi_ > 0


class TestScorecard:
    @pytest.fixture
    def scorecard_data(self):
        rng = np.random.default_rng(0)
        n = 500
        age = pd.cut(rng.integers(18, 70, n), bins=[18, 30, 45, 60, 70], labels=["18-30", "30-45", "45-60", "60-70"])
        income = pd.cut(rng.integers(20, 200, n), bins=[20, 50, 100, 200], labels=["low", "mid", "high"])
        X = pd.DataFrame({"age": age, "income": income})
        y = pd.Series((rng.random(n) > 0.8).astype(int))
        return X, y

    def test_fit_score(self, scorecard_data):
        X, y = scorecard_data
        sc = Scorecard()
        sc.fit(X, y)
        scores = sc.score(X)
        assert scores.shape == (len(X),)

    def test_iv_summary(self, scorecard_data):
        X, y = scorecard_data
        sc = Scorecard().fit(X, y)
        iv = sc.information_value_summary()
        assert "feature" in iv.columns
        assert "iv" in iv.columns
        assert "predictive_power" in iv.columns

    def test_woe_summary(self, scorecard_data):
        X, y = scorecard_data
        sc = Scorecard().fit(X, y)
        woe = sc.woe_summary("age")
        assert "woe" in woe.columns

    def test_woe_summary_invalid_feature(self, scorecard_data):
        X, y = scorecard_data
        sc = Scorecard().fit(X, y)
        with pytest.raises(KeyError):
            sc.woe_summary("nonexistent")
