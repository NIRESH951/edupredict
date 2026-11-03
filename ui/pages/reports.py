"""Report generation: student PDF, class PDF and CSV exports."""
from __future__ import annotations

from datetime import datetime

import streamlit as st

from core.scope import AccessDenied
from services import analytics_service as an
from services import report_service, student_service
from services.academic_service import export_records_csv, list_records
from ui.components import class_filters, empty_state, page_header, show_error, stretch
from ui.session import require_roles, scope


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M")


def render() -> None:
    user = require_roles()
    sc = scope()
    if sc.is_student:
        page_header("My Academic Report", "Download your personal performance report", "PDF")
        _student_report(sc, user, user.get("student_id"))
        return
    page_header("Reports", "Generate student and class performance reports", "PDF & CSV")
    tabs = st.tabs(["Student report", "Class report", "Data exports"])
    with tabs[0]:
        options = student_service.student_options(sc)
        if not options:
            empty_state("No students available.")
        else:
            sid = st.selectbox("Student", list(options), format_func=lambda k: options[k], key="rep_student")
            _student_report(sc, user, sid)
    with tabs[1]:
        _class_report(sc, user)
    with tabs[2]:
        _exports(sc)


def _student_report(sc, user, student_id: str | None) -> None:
    if not student_id:
        st.error("Your account is not linked to a student profile.")
        return
    st.markdown("The report includes the student profile, current-semester predictions and risk, subject "
                "strength analysis, improvement estimates, personalised recommendations, a weekly study plan "
                "and academic history.")
    key = f"pdf_student_{student_id}"
    if st.button("Generate PDF report", type="primary", key=f"gen_{key}", icon=":material/picture_as_pdf:"):
        try:
            with st.spinner("Building report..."):
                st.session_state[key] = report_service.student_report_pdf(student_id, sc, user["full_name"])
        except AccessDenied as exc:
            show_error(exc)
    if st.session_state.get(key):
        st.success("Report ready.")
        st.download_button("Download PDF", st.session_state[key], f"EduPredict_{student_id}_{_stamp()}.pdf",
                           "application/pdf", icon=":material/download:", key=f"dl_{key}", **stretch("download_button"))
        if not sc.is_student:
            recs = list_records(sc, student_id=student_id)
            st.download_button("Download this student's records (CSV)", export_records_csv(recs),
                               f"{student_id}_records.csv", "text/csv", key=f"dlcsv_{key}")


def _class_report(sc, user) -> None:
    dept, sem, sec = class_filters(sc, "rep_class")
    df = an.records_frame(sc, dept, sem, sec)
    n = df["student_id"].nunique() if not df.empty else 0
    st.caption(f"{n} student(s) in this selection. The report contains summary statistics, risk distribution, "
               "subject-wise averages, students needing attention, top performers and a full student list.")
    if n == 0:
        return
    key = f"pdf_class_{dept}_{sem}_{sec}"
    if st.button("Generate class report", type="primary", key=f"gen_{key}", icon=":material/picture_as_pdf:"):
        with st.spinner("Building report..."):
            st.session_state[key] = report_service.class_report_pdf(sc, user["full_name"], dept, sem, sec)
    if st.session_state.get(key):
        st.success("Report ready.")
        label = "_".join(str(x) for x in [dept or "All", f"S{sem}" if sem else "", sec or ""] if x).replace(" ", "")
        st.download_button("Download class report PDF", st.session_state[key], f"EduPredict_Class_{label}_{_stamp()}.pdf",
                           "application/pdf", icon=":material/download:", key=f"dl_{key}")


def _exports(sc) -> None:
    dept, sem, sec = class_filters(sc, "rep_exp")
    df = an.records_frame(sc, dept, sem, sec)
    if df.empty:
        empty_state("No data in this selection.")
        return
    risk = an.student_risk_table(sc, df)
    st.download_button("Academic records with predictions (CSV)", export_records_csv(df),
                       f"records_{_stamp()}.csv", "text/csv", icon=":material/download:")
    st.download_button("Student risk summary (CSV)", risk.to_csv(index=False).encode(), f"risk_summary_{_stamp()}.csv",
                       "text/csv", icon=":material/download:")
    subj = an.subject_averages(df)
    st.download_button("Subject-wise averages (CSV)", subj.to_csv(index=False).encode(),
                       f"subject_averages_{_stamp()}.csv", "text/csv", icon=":material/download:")
