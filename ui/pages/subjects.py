"""Subject catalogue management (admin only)."""
from __future__ import annotations

import streamlit as st

import config
from services import subject_service
from ui.components import empty_state, page_header, show_error, table
from ui.session import require_roles


def render() -> None:
    user = require_roles("admin")
    page_header("Subjects", "Subject catalogue by department and semester", "Admin")
    tabs = st.tabs(["Catalogue", "Add subject", "Edit / delete"])

    with tabs[0]:
        c = st.columns([2, 1.3, 1])
        search = c[0].text_input("Search", placeholder="Code or name", key="sub_search")
        dept = c[1].selectbox("Department", ["All"] + config.DEPARTMENTS, key="sub_dept")
        sem = c[2].selectbox("Semester", ["All"] + config.SEMESTERS, key="sub_sem")
        df = subject_service.list_subjects(dept, sem, search)
        st.caption(f"{len(df)} subject(s)")
        if df.empty:
            empty_state("No subjects found.")
        else:
            table(df.drop(columns=["id"]), {"subject_code": "Code", "subject_name": "Subject", "department": "Department",
                                            "semester": "Sem", "credits": "Credits", "records": "Academic records"},
                  height=480)

    with tabs[1]:
        with st.form("add_subject", clear_on_submit=True):
            c1, c2 = st.columns([1, 2])
            code = c1.text_input("Subject code *", placeholder="AI508")
            name = c2.text_input("Subject name *")
            c3, c4, c5 = st.columns(3)
            dept = c3.selectbox("Department *", config.DEPARTMENTS)
            sem = c4.selectbox("Semester *", config.SEMESTERS, index=4)
            credits = c5.number_input("Credits", 1, 6, 4)
            ok = st.form_submit_button("Add subject", type="primary")
        if ok:
            try:
                subject_service.add_subject(code, name, dept, sem, credits, user)
                st.success(f"Subject {code.upper()} added.")
            except subject_service.SubjectError as exc:
                show_error(exc)

    with tabs[2]:
        df = subject_service.list_subjects()
        if df.empty:
            empty_state("No subjects yet.")
            return
        opts = {r.id: f"{r.subject_code} · {r.subject_name} ({r.department}, Sem {r.semester})" for r in df.itertuples()}
        sid = st.selectbox("Subject", list(opts), format_func=lambda k: opts[k], key="sub_edit_pick")
        row = df[df["id"] == sid].iloc[0]
        with st.form(f"edit_subject_{sid}"):
            c1, c2 = st.columns([1, 2])
            code = c1.text_input("Subject code", row["subject_code"])
            name = c2.text_input("Subject name", row["subject_name"])
            c3, c4, c5 = st.columns(3)
            dept = c3.selectbox("Department", config.DEPARTMENTS, index=config.DEPARTMENTS.index(row["department"]))
            sem = c4.selectbox("Semester", config.SEMESTERS, index=int(row["semester"]) - 1)
            credits = c5.number_input("Credits", 1, 6, int(row["credits"]))
            ok = st.form_submit_button("Save changes", type="primary")
        if ok:
            try:
                subject_service.update_subject(sid, code, name, dept, sem, credits, user)
                st.success("Subject updated.")
                st.rerun()
            except subject_service.SubjectError as exc:
                show_error(exc)
        st.caption("Deleting a subject removes it from the catalogue only; existing academic records keep the "
                   "subject name.")
        if st.button("Delete subject", key=f"del_sub_{sid}"):
            subject_service.delete_subject(sid, user)
            st.success("Subject deleted.")
            st.rerun()
