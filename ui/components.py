"""
Reusable Streamlit building blocks.
"""
from __future__ import annotations

import html
import inspect
from functools import lru_cache

import pandas as pd
import streamlit as st

import config


# --------------------------------------------------------------------------- #
# Streamlit version compatibility for full-width elements
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=None)
def _stretch_kwargs(func_name: str) -> tuple:
    func = getattr(st, func_name)
    params = inspect.signature(func).parameters
    width = params.get("width")
    if width is not None and isinstance(width.default, str):
        return (("width", "stretch"),)
    if "use_container_width" in params:
        return (("use_container_width", True),)
    return ()


def stretch(func_name: str) -> dict:
    return dict(_stretch_kwargs(func_name))


def chart(fig, key: str | None = None) -> None:
    st.plotly_chart(fig, key=key, config={"displaylogo": False}, **stretch("plotly_chart"))


def table(df: pd.DataFrame, column_config: dict | None = None, height: int | None = None, key: str | None = None):
    kwargs = dict(hide_index=True, column_config=column_config or {}, **stretch("dataframe"))
    if height:
        kwargs["height"] = height
    if key:
        kwargs["key"] = key
    st.dataframe(df, **kwargs)


# --------------------------------------------------------------------------- #
# Layout pieces
# --------------------------------------------------------------------------- #
def page_header(title: str, subtitle: str = "", chip: str = "") -> None:
    chip_html = f'<span class="ep-chip">{html.escape(chip)}</span>' if chip else ""
    st.markdown(
        f'<div class="ep-header"><div><h1>{html.escape(title)}</h1>'
        f'<p>{html.escape(subtitle)}</p></div>{chip_html}</div>',
        unsafe_allow_html=True,
    )


def kpi(label: str, value: str, sub: str = "") -> str:
    sub_html = f'<div class="sub">{html.escape(sub)}</div>' if sub else ""
    return (f'<div class="ep-kpi"><div class="lbl">{html.escape(label)}</div>'
            f'<div class="val">{html.escape(value)}</div>{sub_html}</div>')


def kpi_row(items: list[tuple[str, str, str]], compact: bool = False) -> None:
    cols = st.columns(len(items))
    for col, (label, value, sub) in zip(cols, items):
        html_card = kpi(label, value, sub)
        if compact:
            html_card = html_card.replace('class="ep-kpi"', 'class="ep-kpi ep-kpi-sm"')
        col.markdown(html_card, unsafe_allow_html=True)


def risk_badge(level: str | None) -> str:
    if not level or level not in config.RISK_COLORS:
        return '<span class="ep-badge" style="color:#898781">Not assessed</span>'
    color = config.RISK_COLORS[level]
    text_color = "#8a5a00" if level == config.RISK_MODERATE else color
    return (f'<span class="ep-badge" style="color:{text_color}">'
            f'<span class="dot" style="background:{color}"></span>{html.escape(level)}</span>')


def risk_label(level: str | None) -> str:
    """Plain-text risk label with icon, for tables."""
    if not level or level not in config.RISK_ICONS:
        return "— Not assessed"
    return f"{config.RISK_ICONS[level]} {level}"


def recommendation_card(rec) -> None:
    rec = rec if isinstance(rec, dict) else rec.as_dict()
    actions = "".join(f"<li>{html.escape(a)}</li>" for a in rec["actions"])
    subjects = f" · {html.escape(', '.join(rec['subjects']))}" if rec.get("subjects") else ""
    st.markdown(
        f'<div class="ep-card ep-prio-{rec["priority"]}">'
        f'<div class="meta">{html.escape(rec["priority"])} priority · {html.escape(rec["category"])}{subjects}</div>'
        f'<h4>{html.escape(rec["title"])}</h4><div>{html.escape(rec["detail"])}</div>'
        f'<ul>{actions}</ul></div>',
        unsafe_allow_html=True,
    )


def stable_password(slot: str) -> str:
    """A generated temporary password that stays the same across reruns until reset_password_slot()."""
    from core.security import generate_temporary_password

    key = f"_ep_pw_{slot}"
    if key not in st.session_state:
        st.session_state[key] = generate_temporary_password()
    return st.session_state[key]


def reset_password_slot(slot: str) -> None:
    st.session_state.pop(f"_ep_pw_{slot}", None)


def show_error(exc: Exception) -> None:
    for line in str(exc).split("\n"):
        if line.strip():
            st.error(line)


def empty_state(message: str, icon: str = "ℹ️") -> None:
    st.info(message, icon=icon)


def fmt(value, digits: int = 1, suffix: str = "") -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    return f"{float(value):.{digits}f}{suffix}"


def class_filters(scope, key: str, include_section: bool = True, default_all: bool = True) -> tuple:
    """Department / semester / section select boxes limited to classes the user can see."""
    from services.analytics_service import class_options

    classes = class_options(scope)
    depts = ["All"] + sorted(classes["department"].unique().tolist())
    cols = st.columns(3 if include_section else 2)
    dept = cols[0].selectbox("Department", depts, key=f"{key}_dept")
    sub = classes if dept == "All" else classes[classes["department"] == dept]
    sems = ["All"] + sorted(sub["semester"].unique().tolist())
    sem = cols[1].selectbox("Current semester", sems, key=f"{key}_sem")
    sec = "All"
    if include_section:
        sub2 = sub if sem == "All" else sub[sub["semester"] == sem]
        secs = ["All"] + sorted(sub2["section"].unique().tolist())
        sec = cols[2].selectbox("Section", secs, key=f"{key}_sec")
    return (None if dept == "All" else dept, None if sem == "All" else int(sem), None if sec == "All" else sec)


RISK_COLUMN_CONFIG = {
    "predicted_marks": st.column_config.NumberColumn("Predicted", format="%.1f"),
    "final_marks": st.column_config.NumberColumn("Actual final", format="%.1f"),
    "attendance": st.column_config.NumberColumn("Attendance %", format="%.1f"),
    "internal1": st.column_config.NumberColumn("Test 1 /50", format="%.1f"),
    "internal2": st.column_config.NumberColumn("Test 2 /50", format="%.1f"),
    "assignment": st.column_config.NumberColumn("Assignment /20", format="%.1f"),
    "study_hours": st.column_config.NumberColumn("Study hrs/wk", format="%.1f"),
    "previous_gpa": st.column_config.NumberColumn("Prev. GPA", format="%.2f"),
    "risk_confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1, format="%.2f"),
    "internal_avg_pct": st.column_config.NumberColumn("Internal %", format="%.1f"),
}
