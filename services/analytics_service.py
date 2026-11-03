"""
Analytics used by the dashboards (Module 2) and the teacher/student views.
All figures are computed from the SQLite database at request time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from core import database as db
from core.scope import Scope
from services.academic_service import list_records

NOT_ASSESSED = "Not assessed"


def records_frame(scope: Scope, department: str | None = None, current_semester: int | None = None,
                  section: str | None = None) -> pd.DataFrame:
    """All records (every semester) of the students matching the class filters."""
    df = list_records(scope, department=department, section=section)
    if current_semester not in (None, "All") and not df.empty:
        df = df[df["current_semester"] == int(current_semester)]
    return df


def current_records(df: pd.DataFrame) -> pd.DataFrame:
    """Each student's records for their current semester (or latest semester on file)."""
    if df.empty:
        return df
    is_current = df["semester"] == df["current_semester"]
    with_current = set(df.loc[is_current, "student_id"])
    latest = df.groupby("student_id")["semester"].transform("max")
    fallback = ~df["student_id"].isin(with_current) & (df["semester"] == latest)
    return df[is_current | fallback]


def overall_risk(avg_predicted: float, high_subjects: int) -> str:
    """
    Student-level risk from subject-level predictions:
      High     - average predicted marks < 50, or 2+ subjects predicted High Risk
      Moderate - average predicted marks < 70, or exactly 1 subject High Risk
      Low      - otherwise
    """
    if avg_predicted is None or pd.isna(avg_predicted):
        return NOT_ASSESSED
    if avg_predicted < config.MODERATE_RISK_MIN or high_subjects >= 2:
        return config.RISK_HIGH
    if avg_predicted < config.LOW_RISK_MIN or high_subjects == 1:
        return config.RISK_MODERATE
    return config.RISK_LOW


def student_risk_table(scope: Scope, df: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per student summarising their current-semester outlook."""
    df = list_records(scope) if df is None else df
    columns = ["student_id", "reg_no", "full_name", "department", "current_semester", "section", "subjects",
               "avg_attendance", "avg_internal_pct", "avg_predicted", "high_risk_subjects",
               "moderate_risk_subjects", "weakest_subject", "weakest_predicted", "overall_risk"]
    if df.empty:
        return pd.DataFrame(columns=columns)
    cur = current_records(df)
    rows = []
    for sid, g in cur.groupby("student_id"):
        preds = g["predicted_marks"]
        has_pred = preds.notna().any()
        weakest = g.loc[preds.idxmin()] if has_pred else None
        high = int((g["risk_level"] == config.RISK_HIGH).sum())
        avg_pred = float(preds.mean()) if has_pred else np.nan
        first = g.iloc[0]
        rows.append({
            "student_id": sid, "reg_no": first["reg_no"], "full_name": first["full_name"],
            "department": first["department"], "current_semester": int(first["current_semester"]),
            "section": first["section"], "subjects": len(g),
            "avg_attendance": round(float(g["attendance"].mean()), 1),
            "avg_internal_pct": round(float(g["internal_avg_pct"].mean()), 1),
            "avg_predicted": round(avg_pred, 1) if has_pred else np.nan,
            "high_risk_subjects": high,
            "moderate_risk_subjects": int((g["risk_level"] == config.RISK_MODERATE).sum()),
            "weakest_subject": weakest["subject"] if weakest is not None else None,
            "weakest_predicted": round(float(weakest["predicted_marks"]), 1) if weakest is not None else np.nan,
            "overall_risk": overall_risk(avg_pred, high),
        })
    out = pd.DataFrame(rows, columns=columns)
    order = {config.RISK_HIGH: 0, config.RISK_MODERATE: 1, config.RISK_LOW: 2, NOT_ASSESSED: 3}
    out["_order"] = out["overall_risk"].map(order)
    return out.sort_values(["_order", "avg_predicted"]).drop(columns="_order").reset_index(drop=True)


def kpis(scope: Scope, df: pd.DataFrame, risk: pd.DataFrame) -> dict:
    """Headline numbers. Counts of students/teachers/subjects respect the user's scope."""
    if scope.is_admin:
        students = db.table_count("students")
        teachers = db.table_count("teachers")
        subjects = db.table_count("subjects")
    else:
        students = int(risk.shape[0]) if not risk.empty else int(df["student_id"].nunique()) if not df.empty else 0
        teachers = int(db.query_scalar(
            "SELECT COUNT(DISTINCT teacher_id) FROM teacher_assignments WHERE teacher_id = ?",
            [scope.teacher_id], 0)) if scope.is_teacher else 0
        subjects = int(df["subject"].nunique()) if not df.empty else 0
    cur = current_records(df)
    return {
        "students": students,
        "teachers": teachers,
        "subjects": subjects,
        "records": int(len(df)),
        "avg_internal_pct": float(df["internal_avg_pct"].mean()) if not df.empty else 0.0,
        "avg_attendance": float(df["attendance"].mean()) if not df.empty else 0.0,
        "avg_predicted": float(cur["predicted_marks"].mean()) if not cur.empty and cur["predicted_marks"].notna().any() else 0.0,
        "avg_actual": float(df["final_marks"].mean()) if not df.empty and df["final_marks"].notna().any() else 0.0,
        "high_risk": int((risk["overall_risk"] == config.RISK_HIGH).sum()) if not risk.empty else 0,
        "moderate_risk": int((risk["overall_risk"] == config.RISK_MODERATE).sum()) if not risk.empty else 0,
        "low_risk": int((risk["overall_risk"] == config.RISK_LOW).sum()) if not risk.empty else 0,
        "pending_records": int(df["final_marks"].isna().sum()) if not df.empty else 0,
        "low_attendance_records": int((df["attendance"] < config.ATTENDANCE_MINIMUM).sum()) if not df.empty else 0,
    }


def subject_averages(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    g = df.groupby("subject").agg(
        records=("id", "count"),
        avg_final=("final_marks", "mean"),
        avg_predicted=("predicted_marks", "mean"),
        avg_internal_pct=("internal_avg_pct", "mean"),
        avg_attendance=("attendance", "mean"),
    ).reset_index()
    g["avg_marks"] = g["avg_final"].fillna(g["avg_predicted"])
    return g.round(1).sort_values("avg_marks")


def semester_performance(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    return df.groupby("semester").agg(
        records=("id", "count"),
        avg_final=("final_marks", "mean"),
        avg_predicted=("predicted_marks", "mean"),
        avg_internal_pct=("internal_avg_pct", "mean"),
        avg_attendance=("attendance", "mean"),
    ).reset_index().round(1)


def department_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Average actual final marks per department per semester (completed semesters)."""
    done = df.dropna(subset=["final_marks"]) if not df.empty else df
    if done.empty:
        return pd.DataFrame()
    return done.groupby(["department", "semester"]).agg(avg_final=("final_marks", "mean"),
                                                       records=("id", "count")).reset_index().round(1)


def prediction_accuracy(df: pd.DataFrame) -> dict | None:
    """How close stored predictions were for records whose actual marks are now known."""
    both = df.dropna(subset=["final_marks", "predicted_marks"]) if not df.empty else df
    if both.empty:
        return None
    err = both["predicted_marks"] - both["final_marks"]
    actual_risk = both["final_marks"].apply(
        lambda m: config.RISK_LOW if m >= config.LOW_RISK_MIN else
        (config.RISK_MODERATE if m >= config.MODERATE_RISK_MIN else config.RISK_HIGH))
    return {
        "n": int(len(both)),
        "mae": float(err.abs().mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "risk_accuracy": float((actual_risk == both["risk_level"]).mean()),
    }


def class_options(scope: Scope) -> pd.DataFrame:
    """Distinct (department, semester, section) classes visible to the user."""
    clause, params = scope.student_filter("student_id")
    return db.query_df(
        f"""SELECT department, semester, section, COUNT(*) AS students FROM students WHERE {clause}
            GROUP BY department, semester, section ORDER BY department, semester, section""",
        params,
    )
