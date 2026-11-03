"""Dashboards (Module 2): admin analytics, teacher class analytics and the student overview."""
from __future__ import annotations

import streamlit as st

import config
from ml import predictor
from services import analytics_service as an
from services import recommendation_service as rs
from services.report_service import student_insight
from ui import charts, nav
from ui.components import (
    chart,
    class_filters,
    empty_state,
    fmt,
    kpi_row,
    page_header,
    recommendation_card,
    risk_badge,
    risk_label,
    table,
)
from ui.session import require_roles, scope


def render() -> None:
    user = require_roles()
    if user["role"] == "student":
        _student_dashboard(user)
    else:
        _staff_dashboard(user)


# --------------------------------------------------------------------------- #
# Admin / teacher
# --------------------------------------------------------------------------- #
def _staff_dashboard(user: dict) -> None:
    sc = scope()
    is_admin = user["role"] == "admin"
    page_header(
        "Academic Analytics Dashboard" if is_admin else "Class Dashboard",
        "Institution-wide performance, attendance and risk overview" if is_admin
        else "Performance and risk overview for your assigned classes",
        "All data live from the database",
    )
    if not predictor.models_available():
        st.warning("No trained model is active yet, so predicted marks and risk levels are not available. "
                   + ("Train the models on the Model Training page." if is_admin else "Ask the administrator to train the models."))

    dept, sem, sec = class_filters(sc, "dash")
    df = an.records_frame(sc, dept, sem, sec)
    if df.empty:
        empty_state("No academic records match this selection yet.")
        return
    risk = an.student_risk_table(sc, df)
    k = an.kpis(sc, df, risk)
    current = an.current_records(df)

    kpi_row([
        ("Registered students", f"{k['students'] if is_admin and not any([dept, sem, sec]) else len(risk):,}",
         "in current selection" if any([dept, sem, sec]) or not is_admin else "across all departments"),
        ("Teachers" if is_admin else "Your teaching staff ID", f"{k['teachers']}" if is_admin else (user.get('teacher_id') or "—"),
         "registered faculty" if is_admin else "assigned classes below"),
        ("Subjects", f"{k['subjects'] if is_admin else current['subject'].nunique()}",
         "in subject catalogue" if is_admin else "this semester"),
        ("Academic records", f"{k['records']:,}", f"{k['pending_records']:,} awaiting final exam"),
    ])
    st.write("")
    kpi_row([
        ("Average internal marks", fmt(k["avg_internal_pct"], 1, "%"), "average of Test 1 + Test 2"),
        ("Average attendance", fmt(k["avg_attendance"], 1, "%"),
         f"{k['low_attendance_records']:,} records below {config.ATTENDANCE_MINIMUM:.0f}%"),
        ("Average predicted marks", fmt(k["avg_predicted"], 1), "current semester, out of 100"),
        ("Students at high risk", f"{k['high_risk']}", f"+ {k['moderate_risk']} at moderate risk"),
    ])

    st.write("")
    c1, c2 = st.columns([1.35, 1], gap="medium")
    with c1:
        chart(charts.performance_distribution(df), "d_perf")
    with c2:
        chart(charts.risk_distribution(risk), "d_risk")
        acc = an.prediction_accuracy(df)
        if acc:
            st.caption(f"Stored predictions vs actual results on {acc['n']:,} completed records: "
                       f"MAE {acc['mae']:.1f} marks, risk agreement {acc['risk_accuracy']:.0%}. "
                       "These records were also used for training, so see Model Training for unbiased test scores.")

    c3, c4 = st.columns([1.35, 1], gap="medium")
    with c3:
        chart(charts.subject_averages(an.subject_averages(current)), "d_subj")
    with c4:
        chart(charts.attendance_distribution(df), "d_att")

    c5, c6 = st.columns(2, gap="medium")
    with c5:
        chart(charts.semester_performance(an.semester_performance(df)), "d_sem")
    with c6:
        chart(charts.internal_vs_final(df), "d_ivf")

    chart(charts.department_trend(an.department_trend(df)), "d_trend")

    st.markdown("#### Students needing attention")
    attention = risk[risk["overall_risk"].isin([config.RISK_HIGH, config.RISK_MODERATE])].head(15).copy()
    if attention.empty:
        st.success("No students are currently at high or moderate risk in this selection.")
    else:
        attention["overall_risk"] = attention["overall_risk"].map(risk_label)
        table(attention[["reg_no", "full_name", "department", "current_semester", "section", "avg_attendance",
                         "avg_internal_pct", "avg_predicted", "high_risk_subjects", "weakest_subject",
                         "overall_risk"]],
              {"reg_no": "Reg. No.", "full_name": "Name", "department": "Department",
               "current_semester": "Sem", "section": "Sec",
               "avg_attendance": st.column_config.NumberColumn("Attendance %", format="%.1f"),
               "avg_internal_pct": st.column_config.NumberColumn("Internal %", format="%.1f"),
               "avg_predicted": st.column_config.NumberColumn("Avg predicted", format="%.1f"),
               "high_risk_subjects": "High-risk subjects", "weakest_subject": "Weakest subject",
               "overall_risk": "Overall risk"})
        nav.link("risk", "Open full risk analysis", ":material/arrow_forward:")


# --------------------------------------------------------------------------- #
# Student
# --------------------------------------------------------------------------- #
def _student_dashboard(user: dict) -> None:
    sc = scope()
    if not user.get("student_id"):
        st.error("Your account is not linked to a student profile. Please contact the administrator.")
        return
    ins = student_insight(user["student_id"], sc)
    first = user["full_name"].split()[0]
    page_header(f"Welcome back, {first}", f"{ins.student['department']} · Semester {ins.student['semester']} · "
                                          f"Section {ins.student['section']} · {ins.student['reg_no']}",
                "Your personal academic outlook")
    if ins.records.empty:
        empty_state("Your academic records have not been added yet. Please check back later.")
        return

    r = ins.risk_row or {}
    cur = an.current_records(ins.records)
    left, right = st.columns([1, 2.2], gap="large")
    with left:
        st.markdown("##### Overall outlook")
        st.markdown(risk_badge(r.get("overall_risk")), unsafe_allow_html=True)
        st.markdown(f'<div class="ep-big" style="margin-top:.6rem">{fmt(r.get("avg_predicted"))}</div>'
                    '<div class="ep-note">average predicted marks this semester (out of 100)</div>',
                    unsafe_allow_html=True)
        st.write("")
        st.metric("Average attendance", fmt(r.get("avg_attendance"), 1, "%"))
        st.metric("Internal test average", fmt(r.get("avg_internal_pct"), 1, "%"))
        weak = rs.weak_subjects(ins.records)
        st.metric("Weak subjects", f"{len(weak)} of {len(cur)}")
    with right:
        chart(charts.student_subject_outlook(cur), "s_outlook")

    st.markdown("#### Top recommendations for you")
    for rec in ins.recommendations[:3]:
        recommendation_card(rec)
    nav.link("recommendations", "See all recommendations and your study plan", ":material/arrow_forward:")
    chart(charts.student_history(ins.records), "s_hist")
