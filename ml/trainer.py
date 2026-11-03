"""
Model training, evaluation and storage (Module 5).

Regression  (final marks /100):  Random Forest Regressor (primary)  vs Linear Regression (baseline)
Classification (risk category):  Random Forest Classifier (primary) vs Decision Tree Classifier

Models are evaluated on a held-out stratified test split and with k-fold
cross-validation on the training split, then saved with joblib and recorded
in the model_registry table.
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

import config
from core import database as db
from ml.features import FEATURES, TARGET, feature_matrix, risk_labels

MODEL_FILES = {
    "reg_primary": "random_forest_regressor.joblib",
    "reg_baseline": "linear_regression.joblib",
    "clf_primary": "random_forest_classifier.joblib",
    "clf_baseline": "decision_tree_classifier.joblib",
}
ALGORITHM_NAMES = {
    "reg_primary": "Random Forest Regressor",
    "reg_baseline": "Linear Regression",
    "clf_primary": "Random Forest Classifier",
    "clf_baseline": "Decision Tree Classifier",
}


class TrainingError(RuntimeError):
    """User-facing training error."""


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_training_data(include_historical: bool = True) -> tuple[pd.DataFrame, dict]:
    """Labelled rows: academic records with final marks + enabled historical datasets."""
    cols = ", ".join(FEATURES + [TARGET])
    records = db.query_df(f"SELECT {cols} FROM academic_records WHERE {TARGET} IS NOT NULL")
    records["source"] = "academic_records"
    frames = [records]
    if include_historical:
        hist = db.query_df(
            f"""SELECT {", ".join("h." + c for c in FEATURES + [TARGET])}
                FROM historical_records h JOIN datasets d ON d.id = h.dataset_id
                WHERE d.include_in_training = 1"""
        )
        hist["source"] = "historical_datasets"
        frames.append(hist)
    data = pd.concat([f for f in frames if not f.empty], ignore_index=True) if any(
        not f.empty for f in frames) else records
    data = data.dropna(subset=FEATURES + [TARGET])
    counts = data["source"].value_counts().to_dict() if not data.empty else {}
    return data, counts


# --------------------------------------------------------------------------- #
# Model factories
# --------------------------------------------------------------------------- #
def build_models() -> dict:
    return {
        "reg_primary": RandomForestRegressor(
            n_estimators=config.RF_N_ESTIMATORS, min_samples_leaf=2, max_features=0.8,
            random_state=config.RANDOM_STATE, n_jobs=-1,
        ),
        "reg_baseline": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
        "clf_primary": RandomForestClassifier(
            n_estimators=config.RF_N_ESTIMATORS, min_samples_leaf=2, class_weight="balanced",
            random_state=config.RANDOM_STATE, n_jobs=-1,
        ),
        "clf_baseline": DecisionTreeClassifier(
            max_depth=6, min_samples_leaf=5, class_weight="balanced", random_state=config.RANDOM_STATE,
        ),
    }


# --------------------------------------------------------------------------- #
# Metrics helpers
# --------------------------------------------------------------------------- #
def regression_metrics(y_true, y_pred) -> dict:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def classification_metrics(y_true, y_pred) -> dict:
    labels = config.RISK_LEVELS
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision (macro)": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "Recall (macro)": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "F1 (macro)": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "F1 (weighted)": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "report": classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0),
    }


def _feature_stats(X: pd.DataFrame) -> dict:
    stats = {}
    for col in FEATURES:
        s = X[col]
        stats[col] = {
            "mean": float(s.mean()), "median": float(s.median()), "std": float(s.std(ddof=0)),
            "min": float(s.min()), "max": float(s.max()),
            "p25": float(s.quantile(0.25)), "p75": float(s.quantile(0.75)),
        }
    return stats


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train_models(include_historical: bool = True, acting_user: dict | None = None,
                 activate: bool = True, progress=None) -> dict:
    """
    Train, evaluate and save all four models. Returns a summary dict.
    `progress` is an optional callback(fraction, message) for the UI.
    """
    def report(frac: float, msg: str) -> None:
        if progress:
            progress(frac, msg)

    report(0.05, "Loading labelled academic data...")
    data, source_counts = load_training_data(include_historical)
    if len(data) < config.MIN_TRAINING_SAMPLES:
        raise TrainingError(
            f"Not enough labelled records to train: found {len(data)}, need at least "
            f"{config.MIN_TRAINING_SAMPLES}. Add academic records with final examination marks "
            "or upload a historical dataset."
        )

    X = feature_matrix(data)
    y_reg = data[TARGET].astype(float).to_numpy()
    y_clf = risk_labels(y_reg)

    class_counts = pd.Series(y_clf).value_counts()
    stratify = y_clf if class_counts.min() >= 2 and len(class_counts) > 1 else None
    X_train, X_test, yr_train, yr_test, yc_train, yc_test = train_test_split(
        X, y_reg, y_clf, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=stratify,
    )

    models = build_models()
    metrics: dict = {
        "features": FEATURES,
        "n_samples": int(len(data)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "source_counts": {k: int(v) for k, v in source_counts.items()},
        "class_distribution": {k: int(v) for k, v in class_counts.items()},
        "feature_stats": _feature_stats(X_train),
        "target_stats": {"mean": float(np.mean(y_reg)), "std": float(np.std(y_reg))},
    }

    # ---- Regression ------------------------------------------------------- #
    folds = min(config.CV_FOLDS, max(2, len(X_train) // 10))
    kf = KFold(n_splits=folds, shuffle=True, random_state=config.RANDOM_STATE)
    reg_results = {}
    for i, key in enumerate(["reg_primary", "reg_baseline"]):
        report(0.15 + 0.2 * i, f"Training {ALGORITHM_NAMES[key]}...")
        model = models[key]
        cv_r2 = cross_val_score(model, X_train, yr_train, cv=kf, scoring="r2")
        cv_mae = -cross_val_score(model, X_train, yr_train, cv=kf, scoring="neg_mean_absolute_error")
        model.fit(X_train, yr_train)
        pred = np.clip(model.predict(X_test), 0, 100)
        reg_results[key] = {
            "algorithm": ALGORITHM_NAMES[key],
            "test": regression_metrics(yr_test, pred),
            "cv": {"R2_mean": float(cv_r2.mean()), "R2_std": float(cv_r2.std()),
                   "MAE_mean": float(cv_mae.mean()), "MAE_std": float(cv_mae.std()), "folds": folds},
            "test_predictions": [round(float(p), 2) for p in pred],
        }
    reg_results["y_test"] = [round(float(v), 2) for v in yr_test]

    rf_reg = models["reg_primary"]
    lr_model = models["reg_baseline"].named_steps["model"]
    report(0.5, "Computing feature importance...")
    perm = permutation_importance(rf_reg, X_test, yr_test, n_repeats=10,
                                  random_state=config.RANDOM_STATE, n_jobs=-1)
    reg_results["importance"] = {
        "rf_impurity": dict(zip(FEATURES, map(float, rf_reg.feature_importances_))),
        "rf_permutation": dict(zip(FEATURES, map(float, perm.importances_mean))),
        "lr_standardised_coef": dict(zip(FEATURES, map(float, lr_model.coef_))),
    }
    metrics["regression"] = reg_results

    # ---- Classification --------------------------------------------------- #
    min_class = pd.Series(yc_train).value_counts().min()
    cfolds = int(min(config.CV_FOLDS, max(2, min_class)))
    skf = StratifiedKFold(n_splits=cfolds, shuffle=True, random_state=config.RANDOM_STATE)
    clf_results = {}
    for i, key in enumerate(["clf_primary", "clf_baseline"]):
        report(0.6 + 0.15 * i, f"Training {ALGORITHM_NAMES[key]}...")
        model = models[key]
        cv_f1 = cross_val_score(model, X_train, yc_train, cv=skf, scoring="f1_macro")
        cv_acc = cross_val_score(model, X_train, yc_train, cv=skf, scoring="accuracy")
        model.fit(X_train, yc_train)
        pred = model.predict(X_test)
        clf_results[key] = {
            "algorithm": ALGORITHM_NAMES[key],
            "test": classification_metrics(yc_test, pred),
            "cv": {"F1_mean": float(cv_f1.mean()), "F1_std": float(cv_f1.std()),
                   "Accuracy_mean": float(cv_acc.mean()), "Accuracy_std": float(cv_acc.std()), "folds": cfolds},
            "importance": dict(zip(FEATURES, map(float, model.feature_importances_))),
        }
    clf_results["labels"] = config.RISK_LEVELS
    metrics["classification"] = clf_results

    # ---- Save ------------------------------------------------------------- #
    report(0.9, "Saving models...")
    version = datetime.now().strftime("v%Y%m%d-%H%M%S")
    model_dir = config.MODEL_DIR / version
    suffix = 1
    while model_dir.exists():
        suffix += 1
        model_dir = config.MODEL_DIR / f"{version}-{suffix}"
    version = model_dir.name
    model_dir.mkdir(parents=True)

    trained_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for key, filename in MODEL_FILES.items():
        joblib.dump(
            {"model": models[key], "features": FEATURES, "algorithm": ALGORITHM_NAMES[key],
             "task": "regression" if key.startswith("reg") else "classification",
             "version": version, "trained_at": trained_at},
            model_dir / filename,
        )
    (model_dir / "metadata.json").write_text(json.dumps(metrics, indent=2))

    reg_p, clf_p = reg_results["reg_primary"]["test"], clf_results["clf_primary"]["test"]
    with db.get_db() as conn:
        if activate:
            conn.execute("UPDATE model_registry SET is_active = 0")
        conn.execute(
            """INSERT INTO model_registry (version, model_dir, n_samples, n_train, n_test, reg_primary_r2,
                   reg_primary_mae, clf_primary_acc, clf_primary_f1, metrics_json, is_active, trained_by)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [version, model_dir.name, metrics["n_samples"], metrics["n_train"], metrics["n_test"],
             reg_p["R2"], reg_p["MAE"], clf_p["Accuracy"], clf_p["F1 (macro)"], json.dumps(metrics),
             int(activate), acting_user.get("id") if acting_user else None],
        )
    if acting_user:
        db.log_activity(acting_user.get("id"), acting_user.get("username"), "MODEL_TRAINED",
                        f"{version} R2={reg_p['R2']:.3f} Acc={clf_p['Accuracy']:.3f}")
    report(1.0, "Training complete.")
    return {"version": version, "metrics": metrics}


# --------------------------------------------------------------------------- #
# Registry helpers
# --------------------------------------------------------------------------- #
def list_versions() -> pd.DataFrame:
    return db.query_df(
        """SELECT m.version, m.trained_at, m.n_samples, m.n_train, m.n_test,
                  ROUND(m.reg_primary_r2, 3) AS rf_r2, ROUND(m.reg_primary_mae, 2) AS rf_mae,
                  ROUND(m.clf_primary_acc, 3) AS rf_accuracy, ROUND(m.clf_primary_f1, 3) AS rf_f1_macro,
                  m.is_active, u.username AS trained_by
           FROM model_registry m LEFT JOIN users u ON u.id = m.trained_by ORDER BY m.id DESC"""
    )


def model_path(row: dict) -> Path:
    """Model folders are stored relative to config.MODEL_DIR so the project can be moved."""
    path = Path(row["model_dir"])
    return path if path.is_absolute() else config.MODEL_DIR / path


def get_active_version() -> dict | None:
    return db.query_one("SELECT * FROM model_registry WHERE is_active = 1 ORDER BY id DESC LIMIT 1")


def get_version(version: str) -> dict | None:
    return db.query_one("SELECT * FROM model_registry WHERE version = ?", [version])


def activate_version(version: str, acting_user: dict | None = None) -> None:
    row = get_version(version)
    if row is None or not model_path(row).exists():
        raise TrainingError(f"Model files for {version} were not found.")
    with db.get_db() as conn:
        conn.execute("UPDATE model_registry SET is_active = 0")
        conn.execute("UPDATE model_registry SET is_active = 1 WHERE version = ?", [version])
    if acting_user:
        db.log_activity(acting_user.get("id"), acting_user.get("username"), "MODEL_ACTIVATED", version)


def delete_version(version: str, acting_user: dict | None = None) -> None:
    row = get_version(version)
    if row is None:
        raise TrainingError("Model version not found.")
    if row["is_active"]:
        raise TrainingError("The active model cannot be deleted. Activate another version first.")
    shutil.rmtree(model_path(row), ignore_errors=True)
    db.execute("DELETE FROM model_registry WHERE version = ?", [version])
    if acting_user:
        db.log_activity(acting_user.get("id"), acting_user.get("username"), "MODEL_DELETED", version)
