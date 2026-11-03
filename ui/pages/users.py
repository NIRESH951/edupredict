"""User management (admin only): create, activate/deactivate, reset passwords, unlock, delete."""
from __future__ import annotations

import streamlit as st

import config
from services import auth_service, student_service, teacher_service
from ui.components import empty_state, page_header, reset_password_slot, show_error, stable_password, table
from ui.session import require_roles


def render() -> None:
    user = require_roles("admin")
    page_header("User Management", "Login accounts, roles, status and passwords", "Admin")
    tabs = st.tabs(["All users", "Create user", "Manage a user", "Activity log"])

    with tabs[0]:
        c = st.columns([2, 1, 1])
        search = c[0].text_input("Search", placeholder="Username, name or email", key="u_search")
        role = c[1].selectbox("Role", ["All"] + config.ROLES, key="u_role")
        status = c[2].selectbox("Status", ["All", "Active", "Inactive"], key="u_status")
        df = auth_service.list_users(role, search, status)
        if df.empty:
            empty_state("No users found.")
        else:
            df["is_active"] = df["is_active"].map({1: "✅ Active", 0: "⛔ Inactive"})
            df["must_change_password"] = df["must_change_password"].map({1: "Yes", 0: ""})
            table(df, {"id": "ID", "username": "Username", "full_name": "Name", "role": "Role", "email": "Email",
                       "is_active": "Status", "student_id": "Student", "teacher_id": "Teacher",
                       "must_change_password": "Must change pw", "last_login": "Last login",
                       "created_at": "Created"})

    with tabs[1]:
        _create(user)

    with tabs[2]:
        _manage(user)

    with tabs[3]:
        table(auth_service.recent_activity(200), {"created_at": "Time", "username": "User", "action": "Action",
                                                  "details": "Details"}, height=480)


def _create(user: dict) -> None:
    role = st.radio("Account type", config.ROLES, horizontal=True, format_func=lambda r: config.ROLE_LABELS[r],
                    key="new_user_role")
    link_id, default_name, default_email = None, "", ""
    if role == "student":
        pending = student_service.students_without_login()
        if pending.empty:
            st.info("All students already have login accounts.")
            return
        opts = {r.student_id: f"{r.student_id} · {r.reg_no} · {r.full_name}" for r in pending.itertuples()}
        link_id = st.selectbox("Link to student", list(opts), format_func=lambda k: opts[k])
        row = pending[pending["student_id"] == link_id].iloc[0]
        default_name, default_email = row["full_name"], row["email"] or ""
    elif role == "teacher":
        pending = teacher_service.teachers_without_login()
        if pending.empty:
            st.info("All teachers already have login accounts. Add a teacher profile first on the Teachers page.")
            return
        opts = {r.teacher_id: f"{r.teacher_id} · {r.full_name}" for r in pending.itertuples()}
        link_id = st.selectbox("Link to teacher", list(opts), format_func=lambda k: opts[k])
        row = pending[pending["teacher_id"] == link_id].iloc[0]
        default_name, default_email = row["full_name"], row["email"] or ""

    with st.form(f"create_user_{role}_{link_id}", clear_on_submit=True):
        c1, c2 = st.columns(2)
        username = c1.text_input("Username *")
        full_name = c2.text_input("Full name *", default_name)
        email = c1.text_input("Email", default_email)
        password = c2.text_input("Initial password *", stable_password(f"new_{role}_{link_id}"))
        must_change = st.checkbox("Require password change at first login", value=True)
        ok = st.form_submit_button("Create account", type="primary")
    if ok:
        try:
            auth_service.create_user(username, password, role, full_name, email or None,
                                     student_id=link_id if role == "student" else None,
                                     teacher_id=link_id if role == "teacher" else None,
                                     must_change_password=must_change, created_by=user)
            st.success(f"Account `{username}` created. Initial password: `{password}`")
            reset_password_slot(f"new_{role}_{link_id}")
        except auth_service.UserError as exc:
            show_error(exc)


def _manage(user: dict) -> None:
    df = auth_service.list_users()
    opts = {r.id: f"{r.username} · {r.full_name} ({r.role})" for r in df.itertuples()}
    uid = st.selectbox("User", list(opts), format_func=lambda k: opts[k], key="manage_user")
    target = auth_service.get_user(uid)
    status = "Active" if target["is_active"] else "Inactive"
    locked = bool(target["locked_until"])
    st.markdown(f"**Status:** {status} · **Role:** {config.ROLE_LABELS[target['role']]} · "
                f"**Last login:** {target['last_login'] or 'never'}" + (" · 🔒 **Locked**" if locked else ""))

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("##### Status")
        label = "Deactivate account" if target["is_active"] else "Activate account"
        if st.button(label, key=f"toggle_{uid}"):
            try:
                auth_service.set_active(uid, not target["is_active"], user)
                st.success("Status updated.")
                st.rerun()
            except auth_service.UserError as exc:
                show_error(exc)
        if locked and st.button("Unlock account", key=f"unlock_{uid}"):
            auth_service.unlock_user(uid, user)
            st.success("Account unlocked.")
            st.rerun()
    with c2:
        st.markdown("##### Reset password")
        with st.form(f"reset_{uid}", clear_on_submit=True):
            new_pw = st.text_input("New temporary password", stable_password(f"reset_{uid}"))
            ok = st.form_submit_button("Reset password")
        if ok:
            try:
                auth_service.admin_reset_password(uid, new_pw, user)
                st.success(f"Password reset to `{new_pw}`. The user must change it at next login.")
                reset_password_slot(f"reset_{uid}")
            except auth_service.UserError as exc:
                show_error(exc)
    with c3:
        st.markdown("##### Edit details")
        with st.form(f"edit_user_{uid}"):
            name = st.text_input("Full name", target["full_name"])
            email = st.text_input("Email", target["email"] or "")
            ok = st.form_submit_button("Save")
        if ok:
            try:
                auth_service.update_user(uid, name, email, user)
                st.success("Saved.")
            except auth_service.UserError as exc:
                show_error(exc)

    st.divider()
    with st.form(f"delete_user_{uid}"):
        sure = st.checkbox(f"Permanently delete the account `{target['username']}` "
                           "(the linked student/teacher profile is kept)")
        go = st.form_submit_button("Delete account")
    if go:
        if not sure:
            st.error("Please tick the confirmation box.")
        else:
            try:
                auth_service.delete_user(uid, user)
                st.success("Account deleted.")
                st.rerun()
            except auth_service.UserError as exc:
                show_error(exc)
