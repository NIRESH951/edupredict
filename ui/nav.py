"""Per-session registry of navigation pages, so pages can link to each other."""
from __future__ import annotations

import streamlit as st

_KEY = "_ep_pages"


def register(url_path: str, page) -> None:
    st.session_state.setdefault(_KEY, {})[url_path] = page


def clear() -> None:
    st.session_state[_KEY] = {}


def link(url_path: str, label: str, icon: str | None = None) -> None:
    """Render a link to another page if the current user has access to it."""
    page = st.session_state.get(_KEY, {}).get(url_path)
    if page is not None:
        st.page_link(page, label=label, icon=icon)
