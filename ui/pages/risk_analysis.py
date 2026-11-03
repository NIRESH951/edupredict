"""Academic risk analysis for admins and teachers."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config
from ml import predictor
from ml.features import FEATURES
from services import analytics_service as an
from services import recommendation_service as rs
from ui import charts, nav
from ui.components import (
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
from ui.theme import RISK


def render() -> None:
    require_roles("admin", "teacher")
    sc = scope()
    page_header("Risk Analysis", "Identify students who may need academic support before the final examination",
                "Student-level risk from subject predictions")
    if not predictor.models_available():
        st.warning("No trained model is available, so risk levels cannot be shown yet.")
        return
    dept, sem, sec = class_filters(sc, "risk")
    df = an.records_frame(sc, dept, sem, sec)
    if df.empty:
        empty_state("No records for this selection.")
        return
    risk = an.student_risk_table(sc, df)
    counts = risk["overall_risk"].value_counts()
    total = max(len(risk), 1)
    kpi_row([
        ("🔴 High risk", str(int(counts.get(config.RISK_HIGH, 0))),
         f"{counts.get(config.RISK_HIGH, 0) / total:.0%} of students"),
        ("🟠 Moderate risk", str(int(counts.get(config.RISK_MODERATE, 0))),
         f"{counts.get(config.RISK_MODERATE, 0) / total:.0%} of students"),
        ("🟢 Low risk", str(int(counts.get(config.RISK_LOW, 0))), f"{counts.get(config.RISK_LOW, 0) / total:.0%} of students"),
        ("Students analysed", str(len(risk)), f"{int(counts.get(an.NOT_ASSESSED, 0))} not yet assessed"),
    ])
    c1, c2 = st.columns([1, 1.4], gap="medium")
    with c1:
        chart(charts.risk_distribution(risk), "r_dist")
    with c2:
        chart(_subject_risk_chart(an.current_records(df)), "r_subj")

    st.markdown("#### Student risk register")
    f1, f2 = st.columns([2, 2])
    levels = f1.multiselect("Show risk levels", config.RISK_LEVELS, default=[config.RISK_HIGH, config.RISK_MODERATE],
                            key="r_levels")
    search = f2.text_input("Search student", key="r_search")
    view = risk[risk["overall_risk"].isin(levels)]
    if search:
        mask = view["full_name"].str.contains(search, case=False) | view["reg_no"].str.contains(search, case=False)
        view = view[mask]
    show = view.assign(overall_risk=view["overall_risk"].map(risk_label))
    table(show, {"student_id": "Student ID", "reg_no": "Reg. No.", "full_name": "Name", "department": "Department",
                 "current_semester": "Sem", "section": "Sec", "subjects": "Subjects",
                 "avg_attendance": st.column_config.NumberColumn("Attendance %", format="%.1f"),
                 "avg_internal_pct": st.column_config.NumberColumn("Internal %", format="%.1f"),
                 "avg_predicted": st.column_config.NumberColumn("Avg predicted", format="%.1f"),
                 "high_risk_subjects": "High-risk subj.", "moderate_risk_subjects": "Moderate subj.",
                 "weakest_subject": "Weakest subject",
                 "weakest_predicted": st.column_config.NumberColumn("Weakest pred.", format="%.1f"),
                 "overall_risk": "Overall risk"}, height=420)
    st.download_button("Download risk register (CSV)", view.to_csv(index=False).encode(), "risk_register.csv",
                       "text/csv", icon=":material/download:")
    st.caption("Overall risk: **High** if the average predicted mark is below 50 or two or more subjects are "
               "predicted High Risk; **Moderate** if the average is below 70 or one subject is High Risk; "
               "otherwise **Low**.")

    st.markdown("#### Student risk assessment")
    if view.empty:
        empty_state("No students in the selected risk levels.")
        return
    opts = {r.student_id: f"{r.reg_no} · {r.full_name} · {r.overall_risk}" for r in view.itertuples()}
    sid = st.selectbox("Student", list(opts), format_func=lambda k: opts[k], key="r_student")
    _student_detail(df[df["student_id"] == sid], risk[risk["student_id"] == sid].iloc[0].to_dict())


def _subject_risk_chart(cur: pd.DataFrame) -> go.Figure:
    data = cur.dropna(subset=["risk_level"])
    if data.empty:
        return charts._empty("Risk by subject (current semester)")
    pivot = data.pivot_table(index="subject", columns="risk_level", values="id", aggfunc="count", fill_value=0)
    for lvl in config.RISK_LEVELS:
        if lvl not in pivot:
            pivot[lvl] = 0
    pivot = pivot.assign(_h=pivot[config.RISK_HIGH] / pivot.sum(axis=1)).sort_values("_h").tail(15)
    fig = go.Figure()
    for lvl in [config.RISK_HIGH, config.RISK_MODERATE, config.RISK_LOW]:
        fig.add_trace(go.Bar(y=pivot.index, x=pivot[lvl], name=lvl, orientation="h", marker_color=RISK[lvl],
                             marker_line=dict(color="white", width=2),
                             hovertemplate="%{y}<br>" + lvl + ": %{x} students<extra></extra>"))
    fig.update_layout(title="Risk by subject · current semester", barmode="stack",
                      height=max(300, 26 * len(pivot) + 110), xaxis_title="Students", yaxis_title="")
    return fig


def _student_detail(records: pd.DataFrame, row: dict) -> None:
    left, right = st.columns([1, 1.6], gap="large")
    with left:
        st.markdown(f"### {row['full_name']}")
        st.markdown(risk_badge(row["overall_risk"]), unsafe_allow_html=True)
        st.markdown(f"{row['reg_no']} · {row['department']} · Sem {row['current_semester']}{row['section']}")
        c = st.columns(2)
        c[0].metric("Avg predicted", fmt(row["avg_predicted"]))
        c[1].metric("Attendance", fmt(row["avg_attendance"], 1, "%"))
        c[0].metric("Internal avg", fmt(row["avg_internal_pct"], 1, "%"))
        c[1].metric("High-risk subjects", row["high_risk_subjects"])
        st.markdown("##### Suggested teacher actions")
        for a in rs.teacher_actions(row):
            st.markdown(f"- {a}")
        nav.link("recommendations", "Open the student's full recommendations", ":material/arrow_forward:")
    with right:
        cur = an.current_records(records)
        chart(charts.student_subject_outlook(cur), f"r_outlook_{row['student_id']}")
        weakest = cur.dropna(subset=["predicted_marks"]).sort_values("predicted_marks").head(1)
        if not weakest.empty:
            w = weakest.iloc[0]
            exp = predictor.explain({f: w[f] for f in FEATURES})
            st.markdown(f"##### Risk assessment for the weakest subject: {w['subject']}")
            st.markdown(f"Predicted {w['predicted_marks']:.1f}/100 · {risk_label(w['risk_level'])} "
                        f"({w['risk_confidence']:.0%} confidence)")
            for line in exp["summary"][1:]:
                st.markdown(f"- {line}")
