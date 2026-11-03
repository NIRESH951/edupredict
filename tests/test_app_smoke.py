"""Streamlit smoke tests: the login flow and every page for every role render without exceptions."""
import pytest
from streamlit.testing.v1 import AppTest

PAGES = {
    "admin": ["dashboard", "students", "teachers", "users", "subjects", "academic_records", "datasets",
              "model_training", "predictions", "risk_analysis", "recommendations", "reports", "account"],
    "teacher1": ["dashboard", "students", "academic_records", "predictions", "risk_analysis", "recommendations",
                 "reports", "account"],
    "student1": ["dashboard", "student_history", "predictions", "recommendations", "reports", "account"],
}


def _page_runner(module_name: str, username: str):
    import importlib

    import streamlit as st

    from services.auth_service import get_user_by_username, public_user
    from ui import theme
    from ui.session import login

    theme.inject_css()
    if "ep_user" not in st.session_state:
        login(public_user(get_user_by_username(username)))
    importlib.import_module(f"ui.pages.{module_name}").render()


@pytest.mark.parametrize("username,module", [(u, m) for u, ms in PAGES.items() for m in ms])
def test_page_renders(username, module):
    at = AppTest.from_function(_page_runner, args=(module, username), default_timeout=120)
    at.run()
    assert not at.exception, [e.value for e in at.exception]


def test_teacher_cannot_open_admin_pages():
    at = AppTest.from_function(_page_runner, args=("users", "teacher1"), default_timeout=60)
    at.run()
    assert any("permission" in e.value for e in at.error)


def test_login_flow():
    at = AppTest.from_file("../app.py", default_timeout=120)
    at.run()
    assert not at.exception
    at.text_input[0].input("admin")
    at.text_input[1].input("Wrong@123")
    at.button[0].click().run()
    assert any("Invalid username or password" in e.value for e in at.error)
    at.text_input[0].input("admin")
    at.text_input[1].input("Admin@123")
    at.button[0].click().run()
    assert not at.exception
    assert at.session_state["ep_user"]["role"] == "admin"
