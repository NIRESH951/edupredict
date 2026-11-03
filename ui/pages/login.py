"""Login page (Module 1)."""
from __future__ import annotations

import streamlit as st

import config
from services.auth_service import authenticate
from ui.components import stretch
from ui.session import login


def render() -> None:
    _, mid, _ = st.columns([1, 1.25, 1])
    with mid:
        st.markdown(
            f'<div class="ep-login-brand"><div class="logo">🎓</div><h1>{config.APP_NAME}</h1>'
            f'<p>{config.APP_TAGLINE}</p></div>',
            unsafe_allow_html=True,
        )
        reason = st.session_state.pop("ep_logout_reason", None)
        if reason:
            st.warning(reason)

        with st.form("login_form", border=True):
            st.markdown("#### Sign in")
            username = st.text_input("Username", placeholder="e.g. teacher1", autocomplete="username")
            password = st.text_input("Password", type="password", autocomplete="current-password")
            submitted = st.form_submit_button("Sign in", type="primary", **stretch("form_submit_button"))

        if submitted:
            with st.spinner("Verifying credentials..."):
                user, message = authenticate(username, password)
            if user:
                login(user)
                st.rerun()
            else:
                st.error(message)

        with st.expander("Demo accounts"):
            st.markdown(
                "| Role | Username | Password |\n|---|---|---|\n"
                "| Admin | `admin` | `Admin@123` |\n"
                "| Teacher | `teacher1` … `teacher6` | `Teacher@123` |\n"
                "| Student | `student1`, `student2`, `student3` | `Student@123` |"
            )
            st.caption("teacher1 (Dr. Ananya Sharma) teaches AIML semester 5 A/B and 7 A. "
                       "student1, student2 and student3 are in her classes with moderate, high and low "
                       "risk outlooks respectively.")
        st.caption(f"{config.INSTITUTION_NAME} · Accounts are created by the administrator.")
