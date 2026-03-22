"""Tests for modelrisk.credit."""

import numpy as np
import pandas as pd
import pytest

from modelrisk.credit.pd import LogisticPD, RandomForestPD, XGBoostPD, MertonPD
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


class TestRandomForestPD:
    def test_fit_predict(self, binary_classification_data):
        X_train, X_test, y_train, _ = binary_classification_data
        model = RandomForestPD(n_estimators=50, oob_score=True).fit(X_train, y_train)
        proba = model.predict_proba(X_test)
        assert proba.shape == (len(X_test),)
        assert np.all((proba >= 0) & (proba <= 1))

    def test_feature_importance_summary(self, binary_classification_data):
        X_train, _, y_train, _ = binary_classification_data
        model = RandomForestPD(n_estimators=50).fit(X_train, y_train)
        fi = model.feature_importance_summary()
        assert len(fi) == X_train.shape[1]
        assert "importance" in fi.columns
        assert "importance_pct" in fi.columns
        assert abs(fi["importance_pct"].sum() - 100.0) < 1e-6

    def test_oob_score(self, binary_classification_data):
        X_train, _, y_train, _ = binary_classification_data
        model = RandomForestPD(n_estimators=50, oob_score=True).fit(X_train, y_train)
        assert 0 <= model.oob_score_ <= 1

    def test_oob_score_disabled(self, binary_classification_data):
        X_train, _, y_train, _ = binary_classification_data
        model = RandomForestPD(n_estimators=50, oob_score=False).fit(X_train, y_train)
        assert model.oob_score_ is None

    def test_tree_depth_summary(self, binary_classification_data):
        X_train, _, y_train, _ = binary_classification_data
        model = RandomForestPD(n_estimators=50).fit(X_train, y_train)
        depth = model.tree_depth_summary()
        assert set(depth["statistic"]) == {"min", "mean", "median", "max", "std"}
        assert int(depth.set_index("statistic").loc["min", "value"]) >= 1

    def test_permutation_importance(self, binary_classification_data):
        X_train, X_test, y_train, y_test = binary_classification_data
        model = RandomForestPD(n_estimators=50).fit(X_train, y_train)
        perm = model.permutation_importance(X_test, y_test, n_repeats=3)
        assert len(perm) == X_test.shape[1]
        assert "mean_importance" in perm.columns

    def test_partial_dependence(self, binary_classification_data):
        X_train, _, y_train, _ = binary_classification_data
        model = RandomForestPD(n_estimators=50).fit(X_train, y_train)
        pdp = model.partial_dependence(X_train, feature="f0", grid_resolution=20)
        assert "feature_value" in pdp.columns
        assert "mean_pd" in pdp.columns
        assert len(pdp) == 20

    def test_not_fitted_raises(self):
        model = RandomForestPD()
        with pytest.raises(RuntimeError):
            model.predict_proba(np.zeros((5, 3)))

    def test_shared_interface_consistent_with_logistic(self, binary_classification_data):
        """RandomForestPD and LogisticPD feature_importance_summary have the same columns."""
        X_train, X_test, y_train, _ = binary_classification_data
        rf = RandomForestPD(n_estimators=50).fit(X_train, y_train)
        lr = LogisticPD().fit(X_train, y_train)
        assert set(rf.feature_importance_summary().columns) == set(lr.feature_importance_summary().columns)


class TestXGBoostPD:
    def test_fit_predict(self, binary_classification_data):
        X_train, X_test, y_train, _ = binary_classification_data
        model = XGBoostPD(n_estimators=50).fit(X_train, y_train)
        proba = model.predict_proba(X_test)
        assert proba.shape == (len(X_test),)
        assert np.all((proba >= 0) & (proba <= 1))

    def test_feature_importance_summary_gain(self, binary_classification_data):
        X_train, _, y_train, _ = binary_classification_data
        model = XGBoostPD(n_estimators=50).fit(X_train, y_train)
        fi = model.feature_importance_summary(importance_type="gain")
        assert len(fi) == X_train.shape[1]
        assert "importance_type" in fi.columns
        assert (fi["importance_type"] == "gain").all()

    @pytest.mark.parametrize("imp_type", ["gain", "weight", "cover", "total_gain", "total_cover"])
    def test_feature_importance_all_types(self, binary_classification_data, imp_type):
        X_train, _, y_train, _ = binary_classification_data
        model = XGBoostPD(n_estimators=50).fit(X_train, y_train)
        fi = model.feature_importance_summary(importance_type=imp_type)
        assert len(fi) == X_train.shape[1]

    def test_invalid_importance_type(self, binary_classification_data):
        X_train, _, y_train, _ = binary_classification_data
        model = XGBoostPD(n_estimators=50).fit(X_train, y_train)
        with pytest.raises(ValueError):
            model.feature_importance_summary(importance_type="invalid")

    def test_scale_pos_weight_auto(self, binary_classification_data):
        X_train, _, y_train, _ = binary_classification_data
        model = XGBoostPD(n_estimators=50, scale_pos_weight=None).fit(X_train, y_train)
        assert model.scale_pos_weight_used_ is not None
        assert model.scale_pos_weight_used_ > 0

    def test_scale_pos_weight_manual(self, binary_classification_data):
        X_train, _, y_train, _ = binary_classification_data
        model = XGBoostPD(n_estimators=50, scale_pos_weight=2.5).fit(X_train, y_train)
        assert model.scale_pos_weight_used_ == pytest.approx(2.5)

    def test_parameter_summary(self, binary_classification_data):
        X_train, _, y_train, _ = binary_classification_data
        model = XGBoostPD(n_estimators=50).fit(X_train, y_train)
        params = model.parameter_summary()
        assert "n_estimators" in params.index
        assert "learning_rate" in params.index
        assert "scale_pos_weight_used" in params.index

    def test_learning_curve_with_eval_set(self, binary_classification_data):
        X_train, X_test, y_train, y_test = binary_classification_data
        model = XGBoostPD(n_estimators=50).fit(
            X_train, y_train, eval_set=[(X_test, y_test)]
        )
        lc = model.learning_curve_data()
        assert "round" in lc.columns
        assert "value" in lc.columns
        assert len(lc) == 50

    def test_learning_curve_without_eval_set_raises(self, binary_classification_data):
        X_train, _, y_train, _ = binary_classification_data
        model = XGBoostPD(n_estimators=50).fit(X_train, y_train)
        with pytest.raises(RuntimeError):
            model.learning_curve_data()

    def test_not_fitted_raises(self):
        model = XGBoostPD()
        with pytest.raises(RuntimeError):
            model.predict_proba(np.zeros((5, 3)))

    def test_shared_interface_consistent_with_logistic(self, binary_classification_data):
        """XGBoostPD feature_importance_summary has the same core columns as LogisticPD."""
        X_train, _, y_train, _ = binary_classification_data
        xgb = XGBoostPD(n_estimators=50).fit(X_train, y_train)
        lr = LogisticPD().fit(X_train, y_train)
        xgb_cols = set(xgb.feature_importance_summary().columns) - {"importance_type"}
        lr_cols = set(lr.feature_importance_summary().columns)
        assert xgb_cols == lr_cols


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
