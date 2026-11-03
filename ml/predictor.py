"""
Loading the active models, making predictions and explaining them.

Explanations are model-agnostic "what if this factor were typical" contributions:
for every feature we replace the student's value with the training median and
measure how much the prediction changes. A positive contribution means the
student's value pushes the prediction up compared with a typical student.
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd

import config
from ml import trainer
from ml.features import FEATURES, feature_matrix, performance_category, risk_from_marks


class ModelNotAvailable(RuntimeError):
    """Raised when no trained model exists yet."""


@dataclass
class ModelBundle:
    version: str
    reg_primary: object
    reg_baseline: object
    clf_primary: object
    clf_baseline: object
    metrics: dict = field(default_factory=dict)

    @property
    def feature_stats(self) -> dict:
        return self.metrics.get("feature_stats", {})

    @property
    def medians(self) -> dict:
        return {f: self.feature_stats.get(f, {}).get("median", 0.0) for f in FEATURES}


_cache: dict[str, ModelBundle] = {}
_lock = threading.Lock()


def load_bundle(version: str | None = None) -> ModelBundle:
    row = trainer.get_version(version) if version else trainer.get_active_version()
    if row is None:
        raise ModelNotAvailable("No trained model is available yet. An administrator must train the models first.")
    ver = row["version"]
    with _lock:
        if ver in _cache:
            return _cache[ver]
        folder = trainer.model_path(row)
        try:
            parts = {key: joblib.load(folder / fname)["model"] for key, fname in trainer.MODEL_FILES.items()}
        except FileNotFoundError as exc:
            raise ModelNotAvailable(f"Model files for {ver} are missing. Please retrain the models.") from exc
        for model in parts.values():
            # Thread start-up dominates for the small batches used in the app; run inference single-threaded.
            if hasattr(model, "n_jobs"):
                model.n_jobs = 1
        bundle = ModelBundle(version=ver, metrics=json.loads(row["metrics_json"]), **parts)
        _cache[ver] = bundle
        return bundle


def models_available() -> bool:
    try:
        load_bundle()
        return True
    except ModelNotAvailable:
        return False


def clear_cache() -> None:
    with _lock:
        _cache.clear()


def _class_index(model) -> dict[str, int]:
    return {label: i for i, label in enumerate(model.classes_)}


def _risk_score(proba: np.ndarray, classes: dict[str, int]) -> np.ndarray:
    """0-100 score: 100 x (P(High) + 0.5 x P(Moderate))."""
    high = proba[:, classes[config.RISK_HIGH]] if config.RISK_HIGH in classes else 0
    mod = proba[:, classes[config.RISK_MODERATE]] if config.RISK_MODERATE in classes else 0
    return 100 * (high + 0.5 * mod)


def predict_frame(df: pd.DataFrame, bundle: ModelBundle | None = None) -> pd.DataFrame:
    """
    Predict for every row of `df` (which must contain the six feature columns).
    Any `final_marks` column is ignored - it is never used as an input.
    """
    bundle = bundle or load_bundle()
    X = feature_matrix(df)
    out = pd.DataFrame(index=df.index)
    marks = np.clip(bundle.reg_primary.predict(X), 0, 100)
    baseline = np.clip(bundle.reg_baseline.predict(X), 0, 100)
    out["predicted_marks"] = np.round(marks, 2)
    out["baseline_marks"] = np.round(baseline, 2)
    out["performance_category"] = [performance_category(m) for m in marks]
    out["marks_based_risk"] = [risk_from_marks(m) for m in marks]

    proba = bundle.clf_primary.predict_proba(X)
    classes = _class_index(bundle.clf_primary)
    out["risk_level"] = bundle.clf_primary.predict(X)
    out["risk_confidence"] = np.round(proba.max(axis=1), 4)
    for label, col in [(config.RISK_LOW, "prob_low"), (config.RISK_MODERATE, "prob_moderate"),
                       (config.RISK_HIGH, "prob_high")]:
        out[col] = np.round(proba[:, classes[label]], 4) if label in classes else 0.0
    out["risk_score"] = np.round(_risk_score(proba, classes), 1)
    out["dt_risk_level"] = bundle.clf_baseline.predict(X)
    out["model_version"] = bundle.version
    return out


def predict_one(features: dict, bundle: ModelBundle | None = None) -> dict:
    df = pd.DataFrame([{f: float(features[f]) for f in FEATURES}])
    return predict_frame(df, bundle).iloc[0].to_dict()


def explain(features: dict, bundle: ModelBundle | None = None) -> dict:
    """
    Local explanation for a single student-subject record.
    Returns contributions to predicted marks and to the risk score, plus a
    plain-language summary.
    """
    bundle = bundle or load_bundle()
    medians = bundle.medians
    base = pd.DataFrame([{f: float(features[f]) for f in FEATURES}])

    # One row per "set feature f to its median" scenario, predicted in one batch.
    scenarios = pd.concat([base] * len(FEATURES), ignore_index=True)
    for i, f in enumerate(FEATURES):
        scenarios.loc[i, f] = medians[f]
    typical = pd.DataFrame([medians])[FEATURES]

    reg = bundle.reg_primary
    pred = float(np.clip(reg.predict(base[FEATURES])[0], 0, 100))
    scen_marks = np.clip(reg.predict(scenarios[FEATURES]), 0, 100)
    typical_marks = float(np.clip(reg.predict(typical)[0], 0, 100))

    clf = bundle.clf_primary
    classes = _class_index(clf)
    risk_now = float(_risk_score(clf.predict_proba(base[FEATURES]), classes)[0])
    scen_risk = _risk_score(clf.predict_proba(scenarios[FEATURES]), classes)

    rows = []
    for i, f in enumerate(FEATURES):
        rows.append({
            "feature": f,
            "label": config.SHORT_LABELS[f],
            "value": float(features[f]),
            "typical": float(medians[f]),
            "marks_effect": round(pred - float(scen_marks[i]), 2),
            "risk_effect": round(risk_now - float(scen_risk[i]), 2),
        })
    contrib = pd.DataFrame(rows).sort_values("marks_effect", key=np.abs, ascending=False)

    helping = contrib[contrib["marks_effect"] > 0.5].sort_values("marks_effect", ascending=False)
    hurting = contrib[contrib["marks_effect"] < -0.5].sort_values("marks_effect")
    risk_up = contrib[contrib["risk_effect"] > 1].sort_values("risk_effect", ascending=False)

    def describe(r) -> str:
        unit = {"attendance": "%", "internal1": "/50", "internal2": "/50", "assignment": "/20",
                "study_hours": " hrs/week", "previous_gpa": "/10"}[r.feature]
        return f"{r.label} of {r.value:g}{unit} (typical {r.typical:g}{unit})"

    summary = [
        f"Predicted final marks: {pred:.1f}/100, compared with {typical_marks:.1f} for a student "
        f"with typical values on every factor."
    ]
    if len(hurting):
        summary.append("Pulling the prediction down: " + "; ".join(
            f"{describe(r)} → {r.marks_effect:+.1f} marks" for r in hurting.head(3).itertuples()) + ".")
    if len(helping):
        summary.append("Pushing the prediction up: " + "; ".join(
            f"{describe(r)} → {r.marks_effect:+.1f} marks" for r in helping.head(3).itertuples()) + ".")
    if len(risk_up):
        summary.append("Main factors increasing academic risk: " + ", ".join(
            r.label for r in risk_up.head(3).itertuples()) + ".")

    return {
        "predicted_marks": round(pred, 2),
        "typical_marks": round(typical_marks, 2),
        "risk_score": round(risk_now, 1),
        "contributions": contrib.reset_index(drop=True),
        "summary": summary,
        "risk_factors": risk_up["label"].head(3).tolist(),
    }


def what_if(features: dict, changes: dict, bundle: ModelBundle | None = None) -> dict:
    """Predict before/after applying `changes` to `features`."""
    bundle = bundle or load_bundle()
    both = pd.DataFrame([{f: float(features[f]) for f in FEATURES},
                         {f: float({**features, **changes}[f]) for f in FEATURES}])
    preds = predict_frame(both, bundle)
    before, after = preds.iloc[0].to_dict(), preds.iloc[1].to_dict()
    return {"before": before, "after": after,
            "gain": round(after["predicted_marks"] - before["predicted_marks"], 2)}
