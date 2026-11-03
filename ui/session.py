"""
Session management: login state, inactivity timeout and role guards.
"""
from __future__ import annotations

import time

import streamlit as st

import config
from core import database as db
from core.scope import Scope
from services.auth_service import get_user, public_user

_USER = "ep_user"
_LAST_SEEN = "ep_last_seen"


def login(user: dict) -> None:
    st.session_state[_USER] = user
    st.session_state[_LAST_SEEN] = time.time()


def logout(reason: str | None = None) -> None:
    user = st.session_state.get(_USER)
    if user:
        db.log_activity(user.get("id"), user.get("username"), "LOGOUT", reason or "")
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    if reason:
        st.session_state["ep_logout_reason"] = reason


def current_user() -> dict | None:
    """
    Return the logged-in user, re-validated against the database on every run so
    that deactivated or deleted accounts lose access immediately.
    """
    user = st.session_state.get(_USER)
    if not user:
        return None
    last_seen = st.session_state.get(_LAST_SEEN, 0)
    if time.time() - last_seen > config.SESSION_TIMEOUT_MINUTES * 60:
        logout("Your session expired after inactivity. Please sign in again.")
        return None
    fresh = get_user(user["id"])
    if fresh is None or not fresh["is_active"]:
        logout("Your account is no longer active.")
        return None
    st.session_state[_LAST_SEEN] = time.time()
    st.session_state[_USER] = public_user(fresh)
    return st.session_state[_USER]


def scope() -> Scope:
    user = st.session_state.get(_USER)
    if not user:
        raise RuntimeError("Not logged in")
    return Scope.from_user(user)


def require_roles(*roles: str) -> dict:
    """Stop rendering the page unless the current user has one of `roles`."""
    user = current_user()
    if user is None:
        st.error("Please sign in to continue.")
        st.stop()
    if roles and user["role"] not in roles:
        st.error("You do not have permission to view this page.")
        st.stop()
    return user
