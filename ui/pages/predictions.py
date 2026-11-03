"""AI-based performance prediction (Module 5): single, what-if, batch and accuracy views."""
from __future__ import annotations

import pandas as pd
import streamlit as st

import config
from ml import predictor
from ml.features import FEATURES, risk_from_marks
from services import analytics_service as an
from services import prediction_service, student_service
from services.academic_service import list_records
from ui import charts
from ui.components import (
    RISK_COLUMN_CONFIG,
    chart,
    class_filters,
    empty_state,
    fmt,
    kpi_row,
    page_header,
    risk_badge,
    risk_label,
    table,
)
from ui.session import require_roles, scope

SLIDERS = {
    "attendance": (0.0, 100.0, 0.5),
    "internal1": (0.0, 50.0, 0.5),
    "internal2": (0.0, 50.0, 0.5),
    "assignment": (0.0, 20.0, 0.5),
    "study_hours": (0.0, 30.0, 0.5),
    "previous_gpa": (0.0, 10.0, 0.05),
}


def render() -> None:
    user = require_roles()
    sc = scope()
    if not predictor.models_available():
        page_header("Performance Prediction", "Predicted final marks and academic risk")
        st.warning("No trained model is available yet. "
                   + ("Train the models on the Model Training page first." if sc.is_admin
                      else "Please ask the administrator to train the models."))
        return
    bundle = predictor.load_bundle()

    if sc.is_student:
        page_header("My Performance Prediction",
                    "Your expected final examination marks and risk level, with the reasons behind them",
                    f"Model {bundle.version}")
        tabs = st.tabs(["My subjects", "What-if simulator"])
        with tabs[0]:
            _single(sc, user, bundle, student_id=user.get("student_id"))
        with tabs[1]:
            _what_if(sc, bundle, student_id=user.get("student_id"))
        return

    page_header("Performance Prediction",
                "Predict final examination marks (regression) and academic risk (classification)",
                f"Active model {bundle.version}")
    tabs = st.tabs(["Predict for a student", "What-if simulator", "Batch predictions", "Actual vs predicted"])
    with tabs[0]:
        _single(sc, user, bundle)
    with tabs[1]:
        _what_if(sc, bundle)
    with tabs[2]:
        _batch(sc, user)
    with tabs[3]:
        _accuracy(sc)


# --------------------------------------------------------------------------- #
def _model_metrics(bundle) -> None:
    m = bundle.metrics
    reg = m["regression"]["reg_primary"]["test"]
    base = m["regression"]["reg_baseline"]["test"]
    clf = m["classification"]["clf_primary"]["test"]
    with st.expander("Model evaluation metrics (held-out test set)"):
        c = st.columns(4)
        c[0].metric("MAE (Random Forest)", f"{reg['MAE']:.2f} marks",
                    f"{base['MAE'] - reg['MAE']:+.2f} vs Linear Regression", delta_color="normal")
        c[1].metric("RMSE", f"{reg['RMSE']:.2f} marks")
        c[2].metric("R² score", f"{reg['R2']:.3f}", f"{reg['R2'] - base['R2']:+.3f} vs baseline")
        c[3].metric("Risk accuracy (RF classifier)", f"{clf['Accuracy']:.1%}", f"F1 macro {clf['F1 (macro)']:.3f}")
        st.caption(f"Trained on {m['n_train']:,} records and tested on {m['n_test']:,} unseen records. "
                   "Inputs: attendance, two internal tests, assignment, weekly study hours and previous GPA. "
                   "Final exam marks are never used as an input.")


def prediction_panel(features: dict, bundle, actual: float | None = None, key: str = "p") -> None:
    pred = predictor.predict_one(features, bundle)
    exp = predictor.explain(features, bundle)
    marks = pred["predicted_marks"]

    c1, c2 = st.columns([1.25, 1], gap="large")
    with c1:
        st.markdown("##### A. Predicted final examination marks")
        st.markdown(f'<div class="ep-hero"><span class="num">{marks:.1f}</span><span class="unit">/ 100 · '
                    f'Random Forest Regressor</span></div>', unsafe_allow_html=True)
        chart(charts.marks_bullet(marks, actual, "Actual"), f"{key}_bullet")
        diff = pred["baseline_marks"] - marks
        third = ("Actual final marks", fmt(actual), f"prediction error {marks - actual:+.1f}") if actual is not None \
            else ("Marks-based risk band", risk_from_marks(marks), "from predicted marks")
        kpi_row([
            ("Performance category", pred["performance_category"], "from predicted marks"),
            ("Baseline · Linear Regression", fmt(pred["baseline_marks"]), f"{diff:+.1f} vs Random Forest"),
            third,
        ], compact=True)
    with c2:
        st.markdown("##### B. Academic risk classification")
        st.markdown(risk_badge(pred["risk_level"]) +
                    f' <span class="ep-note">confidence {pred["risk_confidence"]:.0%} · '
                    f'Decision Tree says {pred["dt_risk_level"]}</span>', unsafe_allow_html=True)
        chart(charts.probability_bars(pred["prob_low"], pred["prob_moderate"], pred["prob_high"]), f"{key}_prob")

    if pred["risk_level"] != pred["marks_based_risk"]:
        st.info(f"The classifier rates this as **{pred['risk_level']}**, while the predicted marks "
                f"({marks:.1f}) fall in the **{pred['marks_based_risk']}** band. The record is close to a "
                "category boundary – treat it as borderline.", icon="⚖️")

    st.markdown("##### Why the model predicts this")
    for line in exp["summary"]:
        st.markdown(f"- {line}")
    e1, e2 = st.columns(2, gap="medium")
    with e1:
        chart(charts.contributions_chart(exp["contributions"], "marks_effect",
                                         "Effect on predicted marks"), f"{key}_contrib")
    with e2:
        contrib = exp["contributions"].copy()
        contrib["risk_effect"] = -contrib["risk_effect"]  # positive = lowers risk (good), matching left chart
        chart(charts.contributions_chart(contrib, "risk_effect", "Effect on academic risk (positive = lowers risk)"),
              f"{key}_riskc")
    st.caption("Each bar shows how the prediction changes when that factor is replaced by the value of a typical "
               "student (training median), keeping everything else the same.")


def _single(sc, user, bundle, student_id: str | None = None) -> None:
    if student_id is None:
        options = student_service.student_options(sc)
        if not options:
            empty_state("No students available.")
            return
        student_id = st.selectbox("Student", list(options), format_func=lambda k: options[k], key="pred_student")
    recs = list_records(sc, student_id=student_id)
    if recs.empty:
        empty_state("No academic records found for this student.")
        return
    recs = recs.sort_values(["semester", "subject"], ascending=[False, True])
    labels = {r.id: f"Sem {r.semester} · {r.subject}" + ("" if pd.isna(r.final_marks) else " (completed)")
              for r in recs.itertuples()}
    rid = st.selectbox("Subject record", list(labels), format_func=lambda k: labels[k], key=f"pred_rec_{student_id}")
    row = recs[recs["id"] == rid].iloc[0]

    st.markdown("**Inputs used by the model** (all known before the final examination)")
    inputs = pd.DataFrame([{config.SHORT_LABELS[f]: row[f] for f in FEATURES}])
    table(inputs)
    actual = None if pd.isna(row["final_marks"]) else float(row["final_marks"])
    prediction_panel({f: row[f] for f in FEATURES}, bundle, actual, key=f"single_{rid}")

    if not pd.isna(row["predicted_marks"]):
        st.caption(f"Stored prediction: {row['predicted_marks']:.1f} ({row['risk_level']}) from model "
                   f"{row['model_version']} on {row['predicted_at']}.")
    if not sc.is_student:
        if st.button("Save this prediction to the record", key=f"save_pred_{rid}"):
            prediction_service.predict_and_store(sc, [int(rid)], acting_user=user)
            st.success("Prediction saved.")
        hist = prediction_service.prediction_history(int(rid))
        if len(hist) > 1:
            with st.expander("Prediction history for this record"):
                table(hist)
    _model_metrics(bundle)


def _what_if(sc, bundle, student_id: str | None = None) -> None:
    st.caption("Adjust the indicators to see how the predicted marks and risk would change. Nothing is saved.")
    base = dict(bundle.medians)
    if student_id:
        recs = list_records(sc, student_id=student_id)
        cur = an.current_records(recs)
        if not cur.empty:
            labels = {r.id: r.subject for r in cur.itertuples()}
            rid = st.selectbox("Start from my record for", list(labels), format_func=lambda k: labels[k],
                               key="wi_rec")
            base = {f: float(cur[cur["id"] == rid].iloc[0][f]) for f in FEATURES}
    else:
        st.caption("Starting values are those of a typical student in the training data.")
    key = "wi_" + "_".join(f"{base[f]:.1f}" for f in FEATURES)
    cols = st.columns(3)
    values = {}
    for i, f in enumerate(FEATURES):
        lo, hi, step = SLIDERS[f]
        values[f] = cols[i % 3].slider(config.FEATURE_LABELS[f], lo, hi, float(min(max(base[f], lo), hi)), step,
                                       key=f"{key}_{f}")
    before = predictor.predict_one(base, bundle)
    after = predictor.predict_one(values, bundle)
    kpi_row([
        ("Predicted marks", fmt(after["predicted_marks"]),
         f"{after['predicted_marks'] - before['predicted_marks']:+.1f} vs starting values"),
        ("Performance category", after["performance_category"], f"was {before['performance_category']}"),
        ("Risk level", after["risk_level"], f"was {before['risk_level']}"),
        ("Classifier confidence", f"{after['risk_confidence']:.0%}", "Random Forest"),
    ], compact=True)
    chart(charts.marks_bullet(after["predicted_marks"], before["predicted_marks"], "Starting values"), "wi_bullet")
    st.caption("Black bar: prediction with the adjusted values. Blue marker: prediction at the starting values.")


def _batch(sc, user) -> None:
    st.markdown("Run the active models over many records at once and store the results.")
    c1, c2 = st.columns(2)
    if c1.button("Predict records missing a current prediction", type="primary", key="batch_missing"):
        n = prediction_service.predict_and_store(sc, only_missing=True, acting_user=user)
        st.success(f"Generated {n} new prediction(s).")
    if c2.button("Re-predict all records awaiting the final exam", key="batch_pending"):
        n = prediction_service.predict_and_store(sc, only_pending=True, acting_user=user)
        st.success(f"Refreshed {n} prediction(s).")

    st.markdown("##### Latest predictions for upcoming final examinations")
    dept, sem, sec = class_filters(sc, "batch")
    df = an.records_frame(sc, dept, sem, sec)
    pending = df[df["final_marks"].isna()] if not df.empty else df
    if pending.empty:
        empty_state("No records awaiting the final examination in this selection.")
        return
    risk_pick = st.multiselect("Risk level", config.RISK_LEVELS, default=config.RISK_LEVELS, key="batch_risk")
    view = pending[pending["risk_level"].isin(risk_pick)].sort_values("predicted_marks")
    view = view.assign(risk=view["risk_level"].map(risk_label))
    cols = ["reg_no", "full_name", "subject", "semester", "attendance", "internal1", "internal2", "assignment",
            "study_hours", "previous_gpa", "predicted_marks", "performance_category", "risk", "risk_confidence"]
    table(view[cols], {**RISK_COLUMN_CONFIG, "reg_no": "Reg. No.", "full_name": "Name", "subject": "Subject",
                       "semester": "Sem", "performance_category": "Category", "risk": "Risk"}, height=480)
    st.download_button("Download predictions (CSV)", view[cols].to_csv(index=False).encode(), "predictions.csv",
                       "text/csv", icon=":material/download:")


def _accuracy(sc) -> None:
    st.markdown("Compare stored predictions with actual final marks once results are available.")
    df = list_records(sc, status="Final marks available")
    both = df.dropna(subset=["predicted_marks"]) if not df.empty else df
    if both.empty:
        empty_state("No records have both a prediction and an actual final mark yet.")
        return
    acc = an.prediction_accuracy(both)
    c = st.columns(4)
    c[0].metric("Records compared", f"{acc['n']:,}")
    c[1].metric("Mean absolute error", f"{acc['mae']:.2f} marks")
    c[2].metric("RMSE", f"{acc['rmse']:.2f} marks")
    c[3].metric("Risk category agreement", f"{acc['risk_accuracy']:.1%}")
    st.caption("Note: completed records were part of the training data, so these figures are optimistic. "
               "The Model Training page reports scores on records the model never saw.")
    chart(charts.actual_vs_predicted(both["final_marks"].tolist(),
                                     {"Random Forest (stored)": both["predicted_marks"].tolist()},
                                     "Actual vs predicted final marks (stored predictions)"), "acc_scatter")
    both = both.assign(error=(both["predicted_marks"] - both["final_marks"]).round(1),
                       abs_error=(both["predicted_marks"] - both["final_marks"]).abs().round(1))
    st.markdown("##### Largest differences")
    table(both.sort_values("abs_error", ascending=False).head(20)[
        ["reg_no", "full_name", "subject", "semester", "final_marks", "predicted_marks", "error"]],
        {**RISK_COLUMN_CONFIG, "reg_no": "Reg. No.", "full_name": "Name", "subject": "Subject", "semester": "Sem",
         "error": st.column_config.NumberColumn("Predicted − actual", format="%+.1f")})
