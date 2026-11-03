"""Teacher management (admin only): profiles, class assignments and logins."""
from __future__ import annotations

import streamlit as st

import config
from core.security import generate_temporary_password
from services import auth_service, teacher_service
from ui.components import empty_state, page_header, show_error, table
from ui.session import require_roles


def render() -> None:
    user = require_roles("admin")
    page_header("Teacher Management", "Faculty profiles, class assignments and teacher logins", "Admin")
    tabs = st.tabs(["All teachers", "Add teacher", "Edit / delete", "Class assignments"])

    with tabs[0]:
        c = st.columns([2, 1])
        search = c[0].text_input("Search", placeholder="Name, ID or email", key="t_search")
        dept = c[1].selectbox("Department", ["All"] + config.DEPARTMENTS, key="t_dept")
        df = teacher_service.list_teachers(search, dept)
        if df.empty:
            empty_state("No teachers found.")
        else:
            table(df, {"teacher_id": "Teacher ID", "full_name": "Name", "email": "Email", "department": "Department",
                       "designation": "Designation", "phone": "Phone", "login_username": "Login",
                       "classes": "Classes", "students": "Students"})

    with tabs[1]:
        _form("add_teacher", None, user)

    with tabs[2]:
        df = teacher_service.list_teachers()
        if df.empty:
            empty_state("No teachers yet.")
        else:
            opts = {r.teacher_id: f"{r.teacher_id} · {r.full_name}" for r in df.itertuples()}
            tid = st.selectbox("Teacher", list(opts), format_func=lambda k: opts[k], key="t_edit_pick")
            teacher = teacher_service.get_teacher(tid)
            _form(f"edit_{tid}", teacher, user)
            st.divider()
            st.markdown("##### Delete teacher")
            st.caption("Removes the teacher profile, their class assignments and their login account. "
                       "Students and academic records are not affected.")
            with st.form(f"del_{tid}"):
                sure = st.checkbox(f"Yes, delete {teacher['full_name']} ({tid})")
                go = st.form_submit_button("Delete teacher")
            if go:
                if sure:
                    teacher_service.delete_teacher(tid, user)
                    st.success("Teacher deleted.")
                    st.rerun()
                else:
                    st.error("Please tick the confirmation box.")

    with tabs[3]:
        _assignments(user)


def _form(key: str, existing: dict | None, user: dict) -> None:
    e = existing or {}
    with st.form(key, clear_on_submit=existing is None):
        c1, c2 = st.columns(2)
        name = c1.text_input("Full name *", e.get("full_name", ""), placeholder="Dr. Firstname Lastname")
        email = c2.text_input("Email", e.get("email") or "")
        dept = c1.selectbox("Department *", config.DEPARTMENTS,
                            index=config.DEPARTMENTS.index(e["department"]) if e.get("department") in config.DEPARTMENTS else 0)
        desig = c2.selectbox("Designation", config.DESIGNATIONS,
                             index=config.DESIGNATIONS.index(e["designation"]) if e.get("designation") in config.DESIGNATIONS else 3)
        phone = c1.text_input("Phone", e.get("phone") or "")
        create_login = False
        username = ""
        if existing is None:
            create_login = c2.checkbox("Also create a login account", value=True)
            username = c2.text_input("Login username", placeholder="e.g. r.kumar")
        ok = st.form_submit_button("Save changes" if existing else "Add teacher", type="primary")
    if not ok:
        return
    data = {"full_name": name, "email": email, "department": dept, "designation": desig, "phone": phone}
    try:
        if existing:
            teacher_service.update_teacher(existing["teacher_id"], data, user)
            st.success("Teacher updated.")
        else:
            if create_login and not username.strip():
                st.error("Please enter a login username, or untick 'Also create a login account'.")
                return
            tid = teacher_service.add_teacher(data, user)
            msg = f"Teacher {tid} added."
            if create_login:
                pw = generate_temporary_password()
                try:
                    auth_service.create_user(username, pw, "teacher", name, email or None, teacher_id=tid,
                                             must_change_password=True, created_by=user)
                    msg += f" Login `{username}` created with temporary password `{pw}` (must be changed at first login)."
                except auth_service.UserError as exc:
                    msg += f" Login was not created: {exc}"
            st.success(msg)
    except teacher_service.TeacherError as exc:
        show_error(exc)


def _assignments(user: dict) -> None:
    st.caption("A teacher can see and manage every student in the classes (department + semester + section) "
               "assigned to them.")
    teachers = teacher_service.list_teachers()
    if teachers.empty:
        empty_state("Add a teacher first.")
        return
    opts = {r.teacher_id: f"{r.teacher_id} · {r.full_name} ({r.department})" for r in teachers.itertuples()}
    tid = st.selectbox("Teacher", list(opts), format_func=lambda k: opts[k], key="assign_teacher")
    current = teacher_service.list_assignments(tid)
    if current.empty:
        st.info("No classes assigned yet.")
    else:
        table(current[["id", "department", "semester", "section", "students"]],
              {"id": "Assignment #", "department": "Department", "semester": "Semester", "section": "Section",
               "students": "Students"})
    with st.form("add_assignment"):
        c = st.columns([2, 1, 1])
        teacher_dept = teachers.loc[teachers["teacher_id"] == tid, "department"].iloc[0]
        dept = c[0].selectbox("Department", config.DEPARTMENTS, index=config.DEPARTMENTS.index(teacher_dept))
        sem = c[1].selectbox("Semester", config.SEMESTERS, index=4)
        sec = c[2].selectbox("Section", config.SECTIONS)
        ok = st.form_submit_button("Assign class", type="primary")
    if ok:
        try:
            teacher_service.add_assignment(tid, dept, sem, sec, user)
            st.success("Class assigned.")
            st.rerun()
        except teacher_service.TeacherError as exc:
            show_error(exc)
    if not current.empty:
        labels = {r.id: f"{r.department} · Sem {r.semester} · Sec {r.section}" for r in current.itertuples()}
        c1, c2 = st.columns([3, 1])
        aid = c1.selectbox("Remove an assignment", list(labels), format_func=lambda k: labels[k], key="rm_assign")
        c2.write("")
        if c2.button("Remove", key="rm_assign_btn"):
            teacher_service.remove_assignment(aid, user)
            st.success("Assignment removed.")
            st.rerun()
