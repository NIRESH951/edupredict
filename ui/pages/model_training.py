"""Model training and evaluation (admin only)."""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

import config
from core.scope import Scope
from ml import predictor, trainer
from services import prediction_service
from ui import charts
from ui.components import chart, empty_state, page_header, show_error, table
from ui.session import require_roles


def render() -> None:
    user = require_roles("admin")
    page_header("Model Training & Evaluation",
                "Train, compare and manage the regression and risk-classification models", "Admin")
    _train_section(user)
    active = trainer.get_active_version()
    if not active:
        empty_state("No model has been trained yet.")
        return
    metrics = json.loads(active["metrics_json"])
    st.markdown(f"### Active model: `{active['version']}`")
    st.caption(f"Trained {active['trained_at']} on {metrics['n_samples']:,} labelled records "
               f"({metrics['n_train']:,} train / {metrics['n_test']:,} test, stratified "
               f"{int(config.TEST_SIZE * 100)}% hold-out).")
    tabs = st.tabs(["Regression (final marks)", "Classification (risk)", "Feature importance", "Training data",
                    "Model versions"])
    with tabs[0]:
        _regression(metrics)
    with tabs[1]:
        _classification(metrics)
    with tabs[2]:
        _importance(metrics)
    with tabs[3]:
        _data(metrics)
    with tabs[4]:
        _versions(user)


def _train_section(user: dict) -> None:
    data, counts = trainer.load_training_data(include_historical=True)
    with st.container(border=True):
        st.markdown("#### Train new models")
        c = st.columns(4)
        c[0].metric("Labelled academic records", f"{counts.get('academic_records', 0):,}")
        c[1].metric("Historical dataset rows", f"{counts.get('historical_datasets', 0):,}")
        c[2].metric("Minimum required", config.MIN_TRAINING_SAMPLES)
        c[3].metric("Features", len(config.FEATURE_COLUMNS))
        o1, o2 = st.columns(2)
        include_hist = o1.checkbox("Include enabled historical datasets", value=True)
        refresh = o2.checkbox("Refresh predictions for all records after training", value=True)
        st.caption("Regression: Random Forest Regressor (primary) vs Linear Regression (baseline). "
                   "Classification: Random Forest Classifier (primary) vs Decision Tree Classifier. "
                   "Target: actual final examination marks; it is never used as an input feature.")
        if st.button("Train models now", type="primary", icon=":material/model_training:"):
            bar = st.progress(0.0, text="Starting...")
            try:
                result = trainer.train_models(include_historical=include_hist, acting_user=user,
                                              progress=lambda f, m: bar.progress(min(f, 1.0), text=m))
                predictor.clear_cache()
                msg = f"Trained and activated {result['version']}."
                if refresh:
                    n = prediction_service.predict_and_store(Scope.admin(), acting_user=user)
                    msg += f" Refreshed predictions for {n:,} records."
                st.success(msg)
            except trainer.TrainingError as exc:
                show_error(exc)


def _regression(m: dict) -> None:
    reg = m["regression"]
    rows = []
    for key in ("reg_primary", "reg_baseline"):
        r = reg[key]
        rows.append({"Model": r["algorithm"] + (" (primary)" if key == "reg_primary" else " (baseline)"),
                     "MAE": r["test"]["MAE"], "RMSE": r["test"]["RMSE"], "R²": r["test"]["R2"],
                     "CV R² (mean ± sd)": f"{r['cv']['R2_mean']:.3f} ± {r['cv']['R2_std']:.3f}",
                     "CV MAE": f"{r['cv']['MAE_mean']:.2f} ± {r['cv']['MAE_std']:.2f}"})
    table(pd.DataFrame(rows), {"MAE": st.column_config.NumberColumn(format="%.2f"),
                               "RMSE": st.column_config.NumberColumn(format="%.2f"),
                               "R²": st.column_config.NumberColumn(format="%.3f")})
    best = "Random Forest" if reg["reg_primary"]["test"]["R2"] >= reg["reg_baseline"]["test"]["R2"] else "Linear Regression"
    st.caption(f"Lower MAE/RMSE and higher R² are better. {best} performs best on the held-out test set. "
               f"Cross-validation uses {reg['reg_primary']['cv']['folds']} folds on the training split.")
    c1, c2 = st.columns([1.2, 1])
    with c1:
        chart(charts.actual_vs_predicted(reg["y_test"], {
            "Random Forest": reg["reg_primary"]["test_predictions"],
            "Linear Regression": reg["reg_baseline"]["test_predictions"],
        }), "mt_avp")
    with c2:
        chart(charts.metric_comparison(
            ["Random Forest", "Linear Regression"],
            {"MAE": [reg["reg_primary"]["test"]["MAE"], reg["reg_baseline"]["test"]["MAE"]],
             "RMSE": [reg["reg_primary"]["test"]["RMSE"], reg["reg_baseline"]["test"]["RMSE"]]},
            "Error comparison (marks, lower is better)"), "mt_cmp")
        chart(charts.residual_histogram(reg["y_test"], reg["reg_primary"]["test_predictions"], "Random Forest"),
              "mt_resid")


def _classification(m: dict) -> None:
    clf = m["classification"]
    rows = []
    for key in ("clf_primary", "clf_baseline"):
        t = clf[key]["test"]
        rows.append({"Model": clf[key]["algorithm"] + (" (primary)" if key == "clf_primary" else " (baseline)"),
                     "Accuracy": t["Accuracy"], "Precision (macro)": t["Precision (macro)"],
                     "Recall (macro)": t["Recall (macro)"], "F1 (macro)": t["F1 (macro)"],
                     "F1 (weighted)": t["F1 (weighted)"],
                     "CV F1 (mean ± sd)": f"{clf[key]['cv']['F1_mean']:.3f} ± {clf[key]['cv']['F1_std']:.3f}"})
    fmt3 = st.column_config.NumberColumn(format="%.3f")
    table(pd.DataFrame(rows), {k: fmt3 for k in ["Accuracy", "Precision (macro)", "Recall (macro)", "F1 (macro)",
                                                 "F1 (weighted)"]})
    st.caption(f"Risk classes: Low ≥ {config.LOW_RISK_MIN:.0f}, Moderate {config.MODERATE_RISK_MIN:.0f}–"
               f"{config.LOW_RISK_MIN:.0f}, High < {config.MODERATE_RISK_MIN:.0f} final marks. "
               "Both classifiers use balanced class weights.")
    c1, c2 = st.columns(2)
    labels = clf["labels"]
    with c1:
        chart(charts.confusion_matrix_chart(clf["clf_primary"]["test"]["confusion_matrix"], labels,
                                            "Confusion matrix · Random Forest"), "mt_cm1")
    with c2:
        chart(charts.confusion_matrix_chart(clf["clf_baseline"]["test"]["confusion_matrix"], labels,
                                            "Confusion matrix · Decision Tree"), "mt_cm2")
    report = clf["clf_primary"]["test"]["report"]
    per_class = pd.DataFrame([{"Class": lbl, "Precision": report[lbl]["precision"], "Recall": report[lbl]["recall"],
                               "F1": report[lbl]["f1-score"], "Test records": int(report[lbl]["support"])}
                              for lbl in labels if lbl in report])
    st.markdown("##### Per-class results · Random Forest")
    table(per_class, {k: fmt3 for k in ["Precision", "Recall", "F1"]})
    st.caption("Recall on High Risk matters most: it is the share of genuinely at-risk students the model catches.")


def _importance(m: dict) -> None:
    imp = m["regression"]["importance"]
    c1, c2 = st.columns(2)
    with c1:
        chart(charts.feature_importance(imp["rf_permutation"], "Permutation importance · RF regressor (test set)"),
              "fi1")
        chart(charts.feature_importance(m["classification"]["clf_primary"]["importance"],
                                        "Feature importance · RF classifier", charts.SERIES[1]), "fi3")
    with c2:
        chart(charts.feature_importance(imp["rf_impurity"], "Impurity importance · RF regressor"), "fi2")
        chart(charts.feature_importance(imp["lr_standardised_coef"],
                                        "Standardised coefficients · Linear Regression", charts.SERIES[2]), "fi4")
    st.caption("Permutation importance measures how much the test error grows when one feature is shuffled. "
               "Linear Regression coefficients are per one standard deviation of each feature.")


def _data(m: dict) -> None:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Records by source")
        table(pd.DataFrame([{"Source": k.replace("_", " ").title(), "Rows": v}
                            for k, v in m["source_counts"].items()]))
        st.markdown("##### Risk class distribution")
        table(pd.DataFrame([{"Class": k, "Rows": v} for k, v in m["class_distribution"].items()]))
    with c2:
        st.markdown("##### Feature statistics (training split)")
        stats = pd.DataFrame(m["feature_stats"]).T.reset_index().rename(columns={"index": "feature"})
        stats["feature"] = stats["feature"].map(config.SHORT_LABELS)
        table(stats.round(2))


def _versions(user: dict) -> None:
    versions = trainer.list_versions()
    versions["is_active"] = versions["is_active"].map({1: "✅ active", 0: ""})
    table(versions, {"version": "Version", "trained_at": "Trained", "n_samples": "Samples", "n_train": "Train",
                     "n_test": "Test", "rf_r2": "RF R²", "rf_mae": "RF MAE", "rf_accuracy": "RF accuracy",
                     "rf_f1_macro": "RF F1", "is_active": "Status", "trained_by": "By"})
    inactive = versions[versions["is_active"] == ""]["version"].tolist()
    if not inactive:
        return
    c1, c2, c3 = st.columns([2, 1, 1])
    ver = c1.selectbox("Older version", inactive, key="ver_pick")
    c2.write("")
    c3.write("")
    if c2.button("Activate", key="ver_activate"):
        try:
            trainer.activate_version(ver, user)
            predictor.clear_cache()
            n = prediction_service.predict_and_store(Scope.admin(), acting_user=user)
            st.success(f"{ver} is now active. Refreshed {n:,} predictions.")
            st.rerun()
        except trainer.TrainingError as exc:
            show_error(exc)
    if c3.button("Delete", key="ver_delete"):
        try:
            trainer.delete_version(ver, user)
            st.success(f"Deleted {ver}.")
            st.rerun()
        except trainer.TrainingError as exc:
            show_error(exc)
