"""Tests for modelrisk.evaluation."""

import numpy as np
import pytest

from modelrisk.evaluation.classification import ClassificationMetrics
from modelrisk.evaluation.regression import RegressionMetrics
from modelrisk.evaluation.calibration import CalibrationMetrics
from modelrisk.evaluation.explainability import Explainer


class TestClassificationMetrics:
    def test_auc_roc_range(self, predicted_probabilities):
        y_true, y_score = predicted_probabilities
        cm = ClassificationMetrics(y_true, y_score)
        assert 0 <= cm.auc_roc() <= 1

    def test_gini_equals_2auc_minus_1(self, predicted_probabilities):
        y_true, y_score = predicted_probabilities
        cm = ClassificationMetrics(y_true, y_score)
        assert abs(cm.gini() - (2 * cm.auc_roc() - 1)) < 1e-9

    def test_ks_range(self, predicted_probabilities):
        y_true, y_score = predicted_probabilities
        cm = ClassificationMetrics(y_true, y_score)
        ks = cm.ks_statistic()
        assert 0 <= ks["ks"] <= 1

    def test_f1_precision_recall_range(self, predicted_probabilities):
        y_true, y_score = predicted_probabilities
        cm = ClassificationMetrics(y_true, y_score)
        assert 0 <= cm.f1_score() <= 1
        assert 0 <= cm.precision() <= 1
        assert 0 <= cm.recall() <= 1

    def test_brier_score_range(self, predicted_probabilities):
        y_true, y_score = predicted_probabilities
        cm = ClassificationMetrics(y_true, y_score)
        assert 0 <= cm.brier_score() <= 1

    def test_mcc_range(self, predicted_probabilities):
        y_true, y_score = predicted_probabilities
        cm = ClassificationMetrics(y_true, y_score)
        assert -1 <= cm.mcc() <= 1

    def test_lift_at_decile(self, predicted_probabilities):
        y_true, y_score = predicted_probabilities
        cm = ClassificationMetrics(y_true, y_score)
        lift = cm.lift_at_decile(1)
        assert lift >= 0

    def test_summary_shape(self, predicted_probabilities):
        y_true, y_score = predicted_probabilities
        cm = ClassificationMetrics(y_true, y_score)
        summary = cm.summary()
        assert "metric" in summary.columns
        assert "value" in summary.columns
        assert len(summary) >= 10

    def test_roc_curve_data(self, predicted_probabilities):
        y_true, y_score = predicted_probabilities
        cm = ClassificationMetrics(y_true, y_score)
        roc = cm.roc_curve_data()
        assert "fpr" in roc.columns and "tpr" in roc.columns

    def test_single_class_raises(self):
        with pytest.raises(ValueError):
            ClassificationMetrics(np.ones(100), np.random.rand(100))

    def test_perfect_model(self):
        y = np.array([0, 0, 0, 1, 1, 1])
        score = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        cm = ClassificationMetrics(y, score)
        assert cm.auc_roc() == 1.0
        assert cm.gini() == 1.0


class TestRegressionMetrics:
    @pytest.fixture
    def reg_data(self):
        rng = np.random.default_rng(0)
        y_true = rng.uniform(0, 1, 200)
        y_pred = y_true + rng.normal(0, 0.1, 200)
        y_pred = np.clip(y_pred, 0, 1)
        return y_true, y_pred

    def test_rmse_positive(self, reg_data):
        y_true, y_pred = reg_data
        rm = RegressionMetrics(y_true, y_pred)
        assert rm.rmse() >= 0

    def test_mae_leq_rmse(self, reg_data):
        y_true, y_pred = reg_data
        rm = RegressionMetrics(y_true, y_pred)
        assert rm.mae() <= rm.rmse() + 1e-9

    def test_r_squared_range(self, reg_data):
        y_true, y_pred = reg_data
        rm = RegressionMetrics(y_true, y_pred)
        assert rm.r_squared() <= 1.0

    def test_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        rm = RegressionMetrics(y, y)
        assert rm.rmse() < 1e-10
        assert rm.r_squared() == pytest.approx(1.0)

    def test_summary_columns(self, reg_data):
        y_true, y_pred = reg_data
        rm = RegressionMetrics(y_true, y_pred)
        summary = rm.summary()
        assert set(["metric", "value", "interpretation"]).issubset(summary.columns)

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            RegressionMetrics(np.ones(10), np.ones(5))


class TestCalibrationMetrics:
    def test_hosmer_lemeshow_returns_dict(self, predicted_probabilities):
        y_true, y_score = predicted_probabilities
        cal = CalibrationMetrics(y_true, y_score)
        hl = cal.hosmer_lemeshow()
        assert "statistic" in hl
        assert "p_value" in hl
        assert "bins" in hl
        assert 0 <= hl["p_value"] <= 1

    def test_reliability_diagram_data(self, predicted_probabilities):
        y_true, y_score = predicted_probabilities
        cal = CalibrationMetrics(y_true, y_score)
        rd = cal.reliability_diagram_data()
        assert "bin_mean_predicted" in rd.columns
        assert "observed_rate" in rd.columns
        assert (rd["observed_rate"] >= 0).all()
        assert (rd["observed_rate"] <= 1).all()

    def test_expected_vs_actual(self, predicted_probabilities):
        y_true, y_score = predicted_probabilities
        cal = CalibrationMetrics(y_true, y_score)
        eva = cal.expected_vs_actual()
        assert "mean_predicted_pd" in eva.columns
        assert "observed_default_rate" in eva.columns

    def test_ece_range(self, predicted_probabilities):
        y_true, y_score = predicted_probabilities
        cal = CalibrationMetrics(y_true, y_score)
        assert 0 <= cal.expected_calibration_error() <= 1

    def test_summary_columns(self, predicted_probabilities):
        y_true, y_score = predicted_probabilities
        cal = CalibrationMetrics(y_true, y_score)
        summary = cal.summary()
        assert "metric" in summary.columns


class TestExplainer:
    def test_shap_values_shape(self, binary_classification_data, fitted_logistic_pd):
        X_train, X_test, _, _ = binary_classification_data
        explainer = Explainer(
            fitted_logistic_pd,
            feature_names=X_test.columns.tolist(),
            background_data=X_train.values[:50],
        )
        shap_df = explainer.shap_values(X_test.head(10))
        assert shap_df.shape == (10, X_test.shape[1])

    def test_lime_explain_shape(self, binary_classification_data, fitted_logistic_pd):
        X_train, X_test, _, _ = binary_classification_data
        explainer = Explainer(
            fitted_logistic_pd,
            feature_names=X_test.columns.tolist(),
            background_data=X_train,
        )
        result = explainer.lime_explain(X_test.iloc[0], X_train)
        assert "feature" in result.columns
        assert "importance" in result.columns
        assert len(result) == X_test.shape[1]

    def test_permutation_importance(self, binary_classification_data, fitted_logistic_pd):
        X_train, X_test, _, y_test = binary_classification_data
        explainer = Explainer(
            fitted_logistic_pd,
            feature_names=X_test.columns.tolist(),
        )
        perm = explainer.permutation_importance(X_test, y_test, n_repeats=3)
        assert "feature" in perm.columns
        assert "mean_importance" in perm.columns
        assert len(perm) == X_test.shape[1]

    def test_feature_importance_summary(self, binary_classification_data, fitted_logistic_pd):
        X_train, X_test, _, y_test = binary_classification_data
        explainer = Explainer(
            fitted_logistic_pd,
            feature_names=X_test.columns.tolist(),
            background_data=X_train.values[:30],
        )
        summary = explainer.feature_importance_summary(X_test.head(20), y_test.head(20))
        assert "mean_abs_shap" in summary.columns

    def test_shap_available_property(self, fitted_logistic_pd):
        explainer = Explainer(fitted_logistic_pd)
        assert isinstance(explainer.shap_available, bool)
        assert isinstance(explainer.lime_available, bool)
