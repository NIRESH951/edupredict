"""Weak subjects and personalised study recommendations."""
from __future__ import annotations

import streamlit as st

import config
from core.scope import AccessDenied
from services import recommendation_service as rs
from services import student_service
from services.report_service import student_insight
from ui.components import (
    empty_state,
    fmt,
    kpi,
    page_header,
    recommendation_card,
    risk_badge,
    show_error,
    table,
)
from ui.session import require_roles, scope

LEVEL_ICON = {"Critical": "🔴", "Weak": "🟠", "Needs attention": "🟡", "On track": "🔵", "Strong": "🟢"}


def render() -> None:
    user = require_roles()
    sc = scope()
    if sc.is_student:
        page_header("Weak Subjects & Study Recommendations",
                    "Where to focus before the final examination, and how", "Personalised for you")
        student_id = user.get("student_id")
    else:
        page_header("Study Recommendations", "Weak-subject analysis and personalised advice for a student",
                    "Rule-based advice + ML estimates")
        options = student_service.student_options(sc)
        if not options:
            empty_state("No students available.")
            return
        student_id = st.selectbox("Student", list(options), format_func=lambda k: options[k], key="rec_student")
    if not student_id:
        st.error("Your account is not linked to a student profile.")
        return
    try:
        ins = student_insight(student_id, sc)
    except AccessDenied as exc:
        show_error(exc)
        return
    if ins.records.empty:
        empty_state("No academic records yet, so there is nothing to analyse.")
        return

    r = ins.risk_row or {}
    st.markdown(f"**{ins.student['full_name']}** · {ins.student['reg_no']} · overall outlook "
                f"{risk_badge(r.get('overall_risk'))} · average predicted {fmt(r.get('avg_predicted'))}/100",
                unsafe_allow_html=True)

    tabs = st.tabs(["Weak subjects", "Recommendations", "Study plan", "What could improve"])
    with tabs[0]:
        a = ins.assessment
        weak = a[a["level"].isin(["Critical", "Weak"])]
        if weak.empty:
            st.success("No weak subjects this semester. Keep it up!")
        else:
            st.markdown(f"**{len(weak)} weak subject(s)** this semester:")
        for row in a.itertuples():
            with st.container(border=True):
                c1, c2 = st.columns([3, 1])
                c1.markdown(f"**{LEVEL_ICON.get(row.level, '')} {row.subject}** — {row.level}")
                c2.markdown(f"**{fmt(row.marks)}** /100 <span class='ep-note'>({row.marks_source})</span>",
                            unsafe_allow_html=True)
                if row.issues:
                    c1.markdown("; ".join(row.issues))
                else:
                    c1.markdown("No concerns detected.")
        if not ins.past_difficulties.empty:
            st.markdown("##### Earlier subjects with low results")
            table(ins.past_difficulties, {"semester": "Sem", "subject": "Subject",
                                          "final_marks": st.column_config.NumberColumn("Final", format="%.1f"),
                                          "attendance": st.column_config.NumberColumn("Attendance %", format="%.1f"),
                                          "internal_avg_pct": st.column_config.NumberColumn("Internal %", format="%.1f")})

    with tabs[1]:
        for rec in ins.recommendations:
            recommendation_card(rec)
        if not sc.is_student and ins.risk_row:
            st.markdown("##### Suggested teacher actions")
            for act in rs.teacher_actions(ins.risk_row):
                st.markdown(f"- {act}")

    with tabs[2]:
        plan = ins.study_plan
        if plan.empty:
            empty_state("No current-semester subjects to plan for.")
        else:
            c = st.columns(3)
            c[0].metric("Current weekly study", f"{plan['current_hours'].sum():.1f} hrs")
            c[1].metric("Recommended weekly study", f"{plan['recommended_hours'].sum():.1f} hrs",
                        f"+{plan['extra_hours'].sum():.1f} hrs")
            c[2].metric("Per day (6-day week)", f"{plan['recommended_hours'].sum() / 6:.1f} hrs")
            table(plan.assign(status=plan["status"].map(lambda s: f"{LEVEL_ICON.get(s, '')} {s}")),
                  {"subject": "Subject", "status": "Status",
                   "current_hours": st.column_config.NumberColumn("Now (hrs/wk)", format="%.1f"),
                   "recommended_hours": st.column_config.NumberColumn("Target (hrs/wk)", format="%.1f"),
                   "extra_hours": st.column_config.NumberColumn("Extra", format="+%.1f"), "focus": "Focus"})
            st.caption("Extra time goes first to Critical and Weak subjects. Use active recall and spaced "
                       "repetition rather than re-reading.")

    with tabs[3]:
        sc_df = ins.scenarios
        if sc_df is None or sc_df.empty:
            empty_state("No improvement scenarios: either all subjects are strong or the final exams are complete.")
        else:
            st.markdown("Estimated effect of realistic changes before the final exam, using the Random Forest "
                        f"model (attendance → {config.TARGET_ATTENDANCE:.0f}%, study → {config.TARGET_STUDY_HOURS:.0f} "
                        "hrs/week, assignment → 16/20; internal tests unchanged).")
            for row in sc_df.itertuples():
                with st.container(border=True):
                    c1, c2, c3 = st.columns([2.2, 1, 1])
                    c1.markdown(f"**{row.subject}**  \n<span class='ep-note'>{row.changes}</span>",
                                unsafe_allow_html=True)
                    c2.markdown(kpi("Predicted now", fmt(row.current_prediction), row.current_risk),
                                unsafe_allow_html=True)
                    c3.markdown(kpi("If improved", fmt(row.improved_prediction),
                                    f"{row.gain:+.1f} marks · {row.improved_risk}"), unsafe_allow_html=True)
            st.caption("These are model estimates, not guarantees.")
