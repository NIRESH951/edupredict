"""My Account: profile details and password change (all roles)."""
from __future__ import annotations

import streamlit as st

import config
from core.security import password_problems
from services.auth_service import UserError, change_password
from ui.components import page_header, show_error
from ui.session import require_roles


def render() -> None:
    user = require_roles()
    page_header("My Account", "Your profile and password", config.ROLE_LABELS[user["role"]])

    if user.get("must_change_password"):
        st.warning("For security, please set a new password before continuing.", icon="🔐")

    left, right = st.columns([1, 1.2], gap="large")
    with left:
        st.markdown("#### Profile")
        rows = {
            "Full name": user["full_name"],
            "Username": user["username"],
            "Role": config.ROLE_LABELS[user["role"]],
            "Email": user.get("email") or "—",
            "Linked student ID": user.get("student_id") or "—",
            "Linked teacher ID": user.get("teacher_id") or "—",
            "Last login": user.get("last_login") or "—",
            "Account created": user.get("created_at") or "—",
        }
        for k, v in rows.items():
            if v == "—" and k.startswith("Linked"):
                continue
            st.markdown(f"**{k}:** {v}")

    with right:
        st.markdown("#### Change password")
        with st.form("change_pw", clear_on_submit=True):
            current = st.text_input("Current password", type="password")
            new = st.text_input("New password", type="password",
                                help="At least 8 characters with upper- and lower-case letters, a digit and a symbol.")
            confirm = st.text_input("Confirm new password", type="password")
            ok = st.form_submit_button("Update password", type="primary")
        if ok:
            if new != confirm:
                st.error("The new passwords do not match.")
            elif problems := password_problems(new):
                st.error("Password must contain " + ", ".join(problems) + ".")
            else:
                try:
                    change_password(user["id"], current, new)
                    st.success("Password updated successfully.")
                    if user.get("must_change_password"):
                        st.session_state["ep_user"]["must_change_password"] = 0
                        st.rerun()
                except UserError as exc:
                    show_error(exc)
