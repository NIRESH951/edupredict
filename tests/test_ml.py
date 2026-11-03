"""Module 5: training, evaluation, leakage prevention, prediction and explanation."""
import json

import numpy as np
import pandas as pd
import pytest

import config
from core.scope import Scope
from ml import predictor, trainer
from ml.features import FEATURES, performance_category, risk_from_marks
from services import prediction_service
from services.academic_service import list_records


def test_target_is_never_a_feature():
    assert "final_marks" not in FEATURES
    assert FEATURES == ["attendance", "internal1", "internal2", "assignment", "study_hours", "previous_gpa"]


def test_prediction_ignores_final_marks_column():
    row = {"attendance": 80, "internal1": 30, "internal2": 32, "assignment": 15, "study_hours": 6, "previous_gpa": 7}
    a = predictor.predict_frame(pd.DataFrame([{**row, "final_marks": 10}]))
    b = predictor.predict_frame(pd.DataFrame([{**row, "final_marks": 95}]))
    assert a["predicted_marks"].iloc[0] == b["predicted_marks"].iloc[0]


@pytest.mark.parametrize("marks,risk", [(85, "Low Risk"), (70, "Low Risk"), (69.9, "Moderate Risk"),
                                        (50, "Moderate Risk"), (49.9, "High Risk"), (0, "High Risk")])
def test_risk_thresholds(marks, risk):
    assert risk_from_marks(marks) == risk


def test_performance_categories():
    assert performance_category(90) == "Excellent"
    assert performance_category(45) == "Needs Improvement"


def test_active_model_metrics_are_sensible():
    row = trainer.get_active_version()
    m = json.loads(row["metrics_json"])
    rf, lr = m["regression"]["reg_primary"]["test"], m["regression"]["reg_baseline"]["test"]
    assert set(rf) == {"MAE", "RMSE", "R2"}
    assert rf["R2"] > 0.75 and rf["MAE"] < 8
    assert rf["MAE"] <= lr["MAE"] + 0.5
    clf = m["classification"]["clf_primary"]["test"]
    assert clf["Accuracy"] > 0.65
    assert np.array(clf["confusion_matrix"]).sum() == m["n_test"]
    assert (trainer.model_path(row) / "random_forest_regressor.joblib").exists()


def test_predictions_are_bounded_and_stored(admin_scope):
    df = list_records(admin_scope)
    assert df["predicted_marks"].notna().all()
    assert df["predicted_marks"].between(0, 100).all()
    assert set(df["risk_level"].unique()) <= set(config.RISK_LEVELS)
    probs = df[["prob_low", "prob_moderate", "prob_high"]].sum(axis=1)
    assert np.allclose(probs, 1, atol=1e-3)


def test_explanation_structure():
    features = {"attendance": 60, "internal1": 15, "internal2": 14, "assignment": 9, "study_hours": 2,
                "previous_gpa": 5.2}
    exp = predictor.explain(features)
    assert len(exp["contributions"]) == 6
    assert exp["predicted_marks"] < exp["typical_marks"]
    assert exp["contributions"]["marks_effect"].sum() < 0
    assert any("Pulling the prediction down" in s for s in exp["summary"])


def test_better_inputs_predict_higher_marks():
    weak = {"attendance": 60, "internal1": 15, "internal2": 15, "assignment": 8, "study_hours": 2, "previous_gpa": 5}
    strong = {"attendance": 95, "internal1": 45, "internal2": 46, "assignment": 19, "study_hours": 10,
              "previous_gpa": 9}
    w, s = predictor.predict_one(weak), predictor.predict_one(strong)
    assert s["predicted_marks"] > w["predicted_marks"] + 30
    assert w["risk_level"] == "High Risk" and s["risk_level"] == "Low Risk"


def test_retrain_activate_and_delete_versions(admin_user):
    old = trainer.get_active_version()["version"]
    new = trainer.train_models(acting_user=admin_user)["version"]
    assert trainer.get_active_version()["version"] == new
    trainer.activate_version(old, admin_user)
    assert trainer.get_active_version()["version"] == old
    trainer.delete_version(new, admin_user)
    assert trainer.get_version(new) is None
    with pytest.raises(trainer.TrainingError):
        trainer.delete_version(old, admin_user)  # active model is protected


def test_batch_prediction_only_missing(admin_scope):
    assert prediction_service.predict_and_store(admin_scope, only_missing=True) == 0


def test_too_little_data_is_rejected(monkeypatch):
    monkeypatch.setattr(config, "MIN_TRAINING_SAMPLES", 10**7)
    with pytest.raises(trainer.TrainingError):
        trainer.train_models()
