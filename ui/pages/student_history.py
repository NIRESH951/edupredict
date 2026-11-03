"""Student: academic history across semesters."""
from __future__ import annotations

import streamlit as st

from services.academic_service import list_records
from ui import charts
from ui.components import RISK_COLUMN_CONFIG, chart, empty_state, fmt, page_header, table
from ui.session import require_roles, scope


def render() -> None:
    user = require_roles("student")
    sc = scope()
    page_header("Academic History", "Your attendance, internal marks and results across semesters", user["full_name"])
    df = list_records(sc, student_id=user.get("student_id"))
    if df.empty:
        empty_state("No academic records yet.")
        return
    chart(charts.student_history(df), "h_trend")
    for sem in sorted(df["semester"].unique(), reverse=True):
        s = df[df["semester"] == sem]
        done = s["final_marks"].notna().all()
        avg = s["final_marks"].mean() if done else s["predicted_marks"].mean()
        label = f"Semester {sem} · " + (f"average final {fmt(avg)}" if done else f"in progress · average predicted {fmt(avg)}")
        with st.expander(label, expanded=sem == df["semester"].max()):
            cols = ["subject", "attendance", "internal1", "internal2", "assignment", "study_hours", "final_marks",
                    "predicted_marks", "performance_category", "risk_level"]
            table(s[cols], {**RISK_COLUMN_CONFIG, "subject": "Subject", "performance_category": "Category",
                            "risk_level": "Predicted risk"})
