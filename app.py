"""
EduPredict – AI-Based Student Performance Prediction and Academic Improvement System

Run with:   streamlit run app.py
"""
from __future__ import annotations

import inspect
import sqlite3
import threading

import streamlit as st

st.set_page_config(
    page_title="EduPredict",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

import config  # noqa: E402
from core import database as db  # noqa: E402
from core.seed import is_seeded, seed_database  # noqa: E402
from ui import nav, theme  # noqa: E402
from ui.pages import (  # noqa: E402
    account,
    academic_records,
    dashboard,
    datasets,
    login,
    model_training,
    predictions,
    recommendations,
    reports,
    risk_analysis,
    student_history,
    students,
    subjects,
    teachers,
    users,
)
from ui.session import current_user, logout  # noqa: E402

theme.inject_css()


@st.cache_resource(show_spinner=False)
def _setup_lock() -> threading.Lock:
    return threading.Lock()


@st.cache_resource(show_spinner=False)
def _init_schema() -> bool:
    db.init_db()
    return True


def ensure_database() -> None:
    """Create the schema and, on the very first run, seed demo data and train the models."""
    _init_schema()
    try:
        if db.table_count("users") > 0:
            return
    except sqlite3.OperationalError:  # database file was removed while the app was running
        db.init_db()
    with _setup_lock():
        if is_seeded():
            return
        st.markdown("### 🎓 Setting up EduPredict for the first time")
        st.caption("Creating the database, generating a synthetic academic dataset and training the "
                   "machine-learning models. This takes about half a minute and only happens once.")
        bar = st.progress(0.0, text="Starting...")
        seed_database(progress=lambda f, m: bar.progress(min(max(f, 0.0), 1.0), text=m))
    st.rerun()


def page(fn, title: str, icon: str, path: str, default: bool = False):
    p = st.Page(fn, title=title, icon=icon, url_path=path, default=default)
    nav.register(path, p)
    return p


def build_navigation(user: dict) -> dict:
    if user.get("must_change_password"):
        return {"Account": [page(account.render, "Change Password", ":material/lock:", "account", True)]}

    role = user["role"]
    if role == "admin":
        return {
            "Overview": [page(dashboard.render, "Dashboard", ":material/dashboard:", "dashboard", True)],
            "Management": [
                page(students.render, "Students", ":material/school:", "students"),
                page(teachers.render, "Teachers", ":material/co_present:", "teachers"),
                page(users.render, "User Management", ":material/manage_accounts:", "users"),
                page(subjects.render, "Subjects", ":material/menu_book:", "subjects"),
            ],
            "Academic Data": [
                page(academic_records.render, "Academic Records", ":material/edit_note:", "records"),
                page(datasets.render, "Datasets", ":material/dataset:", "datasets"),
            ],
            "AI & Insights": [
                page(model_training.render, "Model Training", ":material/model_training:", "training"),
                page(predictions.render, "Predictions", ":material/insights:", "predictions"),
                page(risk_analysis.render, "Risk Analysis", ":material/warning:", "risk"),
                page(recommendations.render, "Recommendations", ":material/lightbulb:", "recommendations"),
            ],
            "Reports": [page(reports.render, "Reports", ":material/description:", "reports")],
            "Account": [page(account.render, "My Account", ":material/lock:", "account")],
        }
    if role == "teacher":
        return {
            "Overview": [page(dashboard.render, "Class Dashboard", ":material/dashboard:", "dashboard", True)],
            "My Classes": [
                page(students.render, "My Students", ":material/school:", "students"),
                page(academic_records.render, "Academic Records", ":material/edit_note:", "records"),
            ],
            "AI & Insights": [
                page(predictions.render, "Predictions", ":material/insights:", "predictions"),
                page(risk_analysis.render, "Risk Analysis", ":material/warning:", "risk"),
                page(recommendations.render, "Recommendations", ":material/lightbulb:", "recommendations"),
            ],
            "Reports": [page(reports.render, "Reports", ":material/description:", "reports")],
            "Account": [page(account.render, "My Account", ":material/lock:", "account")],
        }
    return {
        "My Progress": [
            page(dashboard.render, "My Dashboard", ":material/dashboard:", "dashboard", True),
            page(student_history.render, "Academic History", ":material/history_edu:", "history"),
            page(predictions.render, "My Prediction", ":material/insights:", "predictions"),
            page(recommendations.render, "Weak Subjects & Tips", ":material/lightbulb:", "recommendations"),
        ],
        "Reports": [page(reports.render, "My Report", ":material/description:", "reports")],
        "Account": [page(account.render, "My Account", ":material/lock:", "account")],
    }


def sidebar(user: dict) -> None:
    with st.sidebar:
        st.markdown(
            f'<div class="ep-user"><div class="name">👤 {user["full_name"]}</div>'
            f'<div class="role">{config.ROLE_LABELS[user["role"]]} · @{user["username"]}</div></div>',
            unsafe_allow_html=True,
        )
        if st.button("Log out", icon=":material/logout:", key="logout_btn", **_stretch()):
            logout()
            st.rerun()
        st.caption(f"{config.APP_NAME} v{config.APP_VERSION} · Decision support only — "
                   "predictions should be reviewed by teachers.")


def _stretch() -> dict:
    from ui.components import stretch
    return stretch("button")


def main() -> None:
    ensure_database()
    user = current_user()
    if user is None:
        login_nav = st.navigation([st.Page(login.render, title="Sign in", icon=":material/login:",
                                           url_path="login", default=True)], position="hidden")
        login_nav.run()
        return
    nav.clear()
    nav_kwargs = {"expanded": True} if "expanded" in inspect.signature(st.navigation).parameters else {}
    navigation = st.navigation(build_navigation(user), **nav_kwargs)
    sidebar(user)
    navigation.run()


main()
