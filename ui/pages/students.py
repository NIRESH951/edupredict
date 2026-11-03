"""Student management (Module 3)."""
from __future__ import annotations

from datetime import date, datetime

import streamlit as st

import config
from core.scope import AccessDenied
from core.security import generate_temporary_password
from services import auth_service, student_service
from services.analytics_service import current_records
from services.report_service import student_insight
from ui import charts
from ui.components import (
    RISK_COLUMN_CONFIG,
    chart,
    empty_state,
    fmt,
    page_header,
    reset_password_slot,
    risk_badge,
    show_error,
    stable_password,
    stretch,
    table,
)
from ui.session import require_roles, scope


def render() -> None:
    user = require_roles("admin", "teacher")
    sc = scope()
    is_admin = user["role"] == "admin"
    page_header("Student Management" if is_admin else "My Students",
                "Add, search, update and manage student profiles" if is_admin
                else "Students in the classes assigned to you",
                "Admin" if is_admin else "Read-only profiles")

    if is_admin:
        tabs = st.tabs(["All students", "Add student", "Edit / delete", "Student profile", "Login accounts"])
        with tabs[0]:
            _list(sc)
        with tabs[1]:
            _add(user)
        with tabs[2]:
            _edit_delete(sc, user)
        with tabs[3]:
            _profile(sc)
        with tabs[4]:
            _logins(user)
    else:
        tabs = st.tabs(["Student list", "Student profile"])
        with tabs[0]:
            _list(sc)
        with tabs[1]:
            _profile(sc)


def _list(sc) -> None:
    c = st.columns([2, 1.3, 1, 1])
    search = c[0].text_input("Search", placeholder="Name, student ID, reg. no. or email", key="stu_search")
    dept = c[1].selectbox("Department", ["All"] + config.DEPARTMENTS, key="stu_dept")
    sem = c[2].selectbox("Semester", ["All"] + config.SEMESTERS, key="stu_sem")
    sec = c[3].selectbox("Section", ["All"] + config.SECTIONS, key="stu_sec")
    df = student_service.list_students(sc, search, dept, sem, sec)
    st.caption(f"{len(df)} student(s)")
    if df.empty:
        empty_state("No students match these filters.")
        return
    table(df, {
        "student_id": "Student ID", "reg_no": "Reg. No.", "full_name": "Full name", "email": "Email",
        "department": "Department", "semester": "Sem", "academic_year": "Academic year", "section": "Sec",
        "enrollment_date": "Enrolled", "login_username": "Login", "records": "Records",
    }, height=min(560, 38 * len(df) + 40))
    st.download_button("Download list (CSV)", df.to_csv(index=False).encode(), "students.csv", "text/csv",
                       icon=":material/download:")


def _student_form(prefix: str, existing: dict | None = None) -> dict | None:
    e = existing or {}
    with st.form(f"{prefix}_form", clear_on_submit=existing is None):
        c1, c2 = st.columns(2)
        reg_no = c1.text_input("Registration number *", e.get("reg_no", ""), placeholder="1EP24AI050")
        full_name = c2.text_input("Full name *", e.get("full_name", ""))
        email = c1.text_input("Email", e.get("email") or "")
        dept_idx = config.DEPARTMENTS.index(e["department"]) if e.get("department") in config.DEPARTMENTS else 0
        department = c2.selectbox("Department *", config.DEPARTMENTS, index=dept_idx)
        c3, c4, c5, c6 = st.columns(4)
        semester = c3.selectbox("Semester *", config.SEMESTERS, index=int(e.get("semester", 1)) - 1)
        section_opts = sorted(set(config.SECTIONS) | ({e["section"]} if e.get("section") else set()))
        section = c4.selectbox("Section *", section_opts,
                               index=section_opts.index(e["section"]) if e.get("section") in section_opts else 0)
        year_default = e.get("academic_year") or f"{date.today().year if date.today().month >= 7 else date.today().year - 1}-" \
                                                 f"{(date.today().year + 1 if date.today().month >= 7 else date.today().year) % 100:02d}"
        academic_year = c5.text_input("Academic year *", year_default, help="Format: 2026-27")
        enrol_default = datetime.strptime(e["enrollment_date"], "%Y-%m-%d").date() if e.get("enrollment_date") else date.today()
        enrollment = c6.date_input("Enrollment date *", enrol_default, max_value=date.today())
        submitted = st.form_submit_button("Save student" if existing else "Add student", type="primary")
    if not submitted:
        return None
    return {"reg_no": reg_no, "full_name": full_name, "email": email, "department": department,
            "semester": semester, "section": section, "academic_year": academic_year, "enrollment_date": enrollment}


def _add(user) -> None:
    st.caption(f"The student ID is generated automatically (next: **{student_service.next_student_id()}**). "
               "Registration numbers and emails must be unique.")
    data = _student_form("add")
    if data:
        try:
            sid = student_service.add_student(data, user)
            st.success(f"Student {sid} added. You can create their login on the *Login accounts* tab.")
        except student_service.StudentError as exc:
            show_error(exc)


def _pick_student(sc, key: str) -> str | None:
    options = student_service.student_options(sc)
    if not options:
        empty_state("No students available.")
        return None
    return st.selectbox("Select student", list(options.keys()), format_func=lambda k: options[k], key=key)


def _edit_delete(sc, user) -> None:
    sid = _pick_student(sc, "edit_pick")
    if not sid:
        return
    student = student_service.get_student(sid, sc)
    st.markdown(f"##### Edit {student['full_name']} ({sid})")
    data = _student_form(f"edit_{sid}", student)
    if data:
        try:
            student_service.update_student(sid, data, sc, user)
            st.success("Student details updated.")
            st.rerun()
        except (student_service.StudentError, AccessDenied) as exc:
            show_error(exc)

    st.divider()
    st.markdown("##### Delete student")
    impact = student_service.deletion_impact(sid)
    accounts = ", ".join(a["username"] for a in impact["accounts"]) or "none"
    st.warning(f"Deleting **{student['full_name']}** will permanently remove {impact['records']} academic record(s), "
               f"{impact['predictions']} prediction(s) and login account(s): {accounts}. This cannot be undone.",
               icon="⚠️")
    with st.form(f"delete_{sid}"):
        confirm = st.text_input(f"Type the registration number **{student['reg_no']}** to confirm")
        sure = st.checkbox("I understand that this deletion is permanent")
        go = st.form_submit_button("Delete student permanently")
    if go:
        if not sure:
            st.error("Please tick the confirmation box.")
        else:
            try:
                student_service.delete_student(sid, confirm, sc, user)
                st.success(f"Student {sid} and all related data were deleted.")
                st.rerun()
            except (student_service.StudentError, AccessDenied) as exc:
                show_error(exc)


def _profile(sc) -> None:
    sid = _pick_student(sc, "profile_pick")
    if not sid:
        return
    try:
        ins = student_insight(sid, sc)
    except AccessDenied as exc:
        show_error(exc)
        return
    s, r = ins.student, ins.risk_row or {}
    left, right = st.columns([1, 2], gap="large")
    with left:
        st.markdown(f"### {s['full_name']}")
        st.markdown(risk_badge(r.get("overall_risk")), unsafe_allow_html=True)
        st.markdown(
            f"**Student ID:** {s['student_id']}  \n**Reg. No.:** {s['reg_no']}  \n**Email:** {s.get('email') or '—'}  \n"
            f"**Department:** {s['department']}  \n**Semester / Section:** {s['semester']} / {s['section']}  \n"
            f"**Academic year:** {s['academic_year']}  \n**Enrolled:** {s['enrollment_date']}  \n"
            f"**Login account:** {s.get('login_username') or 'not assigned'}"
        )
        c1, c2 = st.columns(2)
        c1.metric("Avg predicted", fmt(r.get("avg_predicted")))
        c2.metric("Attendance", fmt(r.get("avg_attendance"), 1, "%"))
        c1.metric("Internal avg", fmt(r.get("avg_internal_pct"), 1, "%"))
        c2.metric("High-risk subjects", r.get("high_risk_subjects", 0))
    with right:
        chart(charts.student_subject_outlook(current_records(ins.records)), f"prof_{sid}")
    if not ins.records.empty:
        st.markdown("##### All academic records")
        cols = ["semester", "subject", "attendance", "internal1", "internal2", "assignment", "study_hours",
                "previous_gpa", "final_marks", "predicted_marks", "risk_level"]
        table(ins.records[cols], {**RISK_COLUMN_CONFIG, "semester": "Sem", "subject": "Subject",
                                  "risk_level": "Predicted risk"})


def _logins(user) -> None:
    st.markdown("##### Assign a login account to a student")
    pending = student_service.students_without_login()
    if pending.empty:
        st.success("Every student already has a login account.")
        return
    st.caption(f"{len(pending)} student(s) do not have a login yet. Usernames default to the registration number.")
    opts = {r.student_id: f"{r.student_id} · {r.reg_no} · {r.full_name}" for r in pending.itertuples()}
    sid = st.selectbox("Student", list(opts.keys()), format_func=lambda k: opts[k], key="login_pick")
    row = pending[pending["student_id"] == sid].iloc[0]
    with st.form("assign_login"):
        c1, c2 = st.columns(2)
        username = c1.text_input("Username", row["reg_no"].lower())
        password = c2.text_input("Temporary password", stable_password(f"stu_{sid}"),
                                 help="The student must change it at first login.")
        ok = st.form_submit_button("Create login", type="primary")
    if ok:
        try:
            auth_service.create_user(username, password, "student", row["full_name"], row["email"],
                                     student_id=sid, must_change_password=True, created_by=user)
            st.success(f"Login created for {row['full_name']}. Username: `{username}` · "
                       f"Temporary password: `{password}` — share it securely.")
            reset_password_slot(f"stu_{sid}")
        except auth_service.UserError as exc:
            show_error(exc)

    with st.expander("Create logins for all students without one (bulk)"):
        st.caption("Creates accounts with username = registration number and a random temporary password per "
                   "student. Download the credential sheet immediately; passwords are not stored in readable form.")
        if st.button("Generate accounts for all", key="bulk_logins"):
            created = []
            bar = st.progress(0.0)
            for i, r in enumerate(pending.itertuples(), start=1):
                pw = generate_temporary_password()
                try:
                    auth_service.create_user(r.reg_no.lower(), pw, "student", r.full_name, r.email,
                                             student_id=r.student_id, must_change_password=True, created_by=user)
                    created.append({"student_id": r.student_id, "reg_no": r.reg_no, "name": r.full_name,
                                    "username": r.reg_no.lower(), "temporary_password": pw})
                except auth_service.UserError as exc:
                    created.append({"student_id": r.student_id, "reg_no": r.reg_no, "name": r.full_name,
                                    "username": "-", "temporary_password": f"FAILED: {exc}"})
                bar.progress(i / len(pending))
            import pandas as pd
            sheet = pd.DataFrame(created)
            st.session_state["bulk_sheet"] = sheet.to_csv(index=False).encode()
            st.success(f"Processed {len(sheet)} student(s).")
        if st.session_state.get("bulk_sheet"):
            st.download_button("Download credential sheet (CSV)", st.session_state["bulk_sheet"],
                               "student_credentials.csv", "text/csv", icon=":material/download:",
                               **stretch("download_button"))
