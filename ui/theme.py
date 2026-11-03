"""
Visual theme: page CSS and the Plotly chart template.

Chart colours follow a fixed categorical order (never cycled) and risk levels
use reserved status colours that are always paired with a text label.
"""
from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

import config

# Categorical slots, assigned in this fixed order.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
RISK = config.RISK_COLORS

CSS = """
<style>
:root { --ep-primary:#2a5bd7; --ep-ink:#0b0b0b; --ep-ink2:#52514e; --ep-muted:#898781;
        --ep-line:rgba(11,11,11,0.10); --ep-card:#ffffff; }
.block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1320px; }
h1, h2, h3 { letter-spacing: -0.01em; }
.ep-header { display:flex; align-items:flex-end; justify-content:space-between; gap:1rem;
             border-bottom:1px solid var(--ep-line); padding-bottom:.75rem; margin-bottom:1.1rem; }
.ep-header h1 { font-size:1.65rem; margin:0; padding:0; line-height:1.2; }
.ep-header p { color:var(--ep-ink2); margin:.25rem 0 0 0; font-size:.95rem; }
.ep-chip { font-size:.78rem; color:var(--ep-ink2); background:#eef1f7; border-radius:999px;
           padding:.2rem .7rem; white-space:nowrap; }
.ep-kpi { background:var(--ep-card); border:1px solid var(--ep-line); border-radius:12px;
          padding:.85rem 1rem .8rem 1rem; height:100%; }
.ep-kpi .lbl { color:var(--ep-ink2); font-size:.8rem; margin-bottom:.15rem; }
.ep-kpi .val { color:var(--ep-ink); font-size:1.6rem; font-weight:650; line-height:1.15; }
.ep-kpi .sub { color:var(--ep-muted); font-size:.76rem; margin-top:.15rem; }
.ep-kpi-sm .val { font-size:1.15rem; }
.ep-hero { display:flex; align-items:baseline; gap:.4rem; margin:.1rem 0 .2rem 0; }
.ep-hero .num { font-size:2.5rem; font-weight:700; line-height:1; color:var(--ep-ink); }
.ep-hero .unit { color:var(--ep-ink2); font-size:1rem; }
.ep-badge { display:inline-flex; align-items:center; gap:.35rem; font-weight:600; font-size:.82rem;
            padding:.18rem .6rem; border-radius:999px; border:1px solid currentColor; background:#fff; }
.ep-badge .dot { width:.55rem; height:.55rem; border-radius:50%; display:inline-block; }
.ep-card { background:var(--ep-card); border:1px solid var(--ep-line); border-radius:12px; padding:1rem 1.1rem;
           margin-bottom:.75rem; }
.ep-card h4 { margin:0 0 .35rem 0; font-size:1rem; }
.ep-card .meta { color:var(--ep-ink2); font-size:.8rem; margin-bottom:.35rem; }
.ep-card ul { margin:.35rem 0 0 1.1rem; padding:0; }
.ep-card li { margin-bottom:.15rem; font-size:.9rem; }
.ep-prio-High { border-left:4px solid #d03b3b; }
.ep-prio-Medium { border-left:4px solid #fab219; }
.ep-prio-Low { border-left:4px solid #0ca30c; }
.ep-big { font-size:2.6rem; font-weight:700; line-height:1; color:var(--ep-ink); }
.ep-note { color:var(--ep-ink2); font-size:.82rem; }
.ep-user { background:#fff; border:1px solid var(--ep-line); border-radius:12px; padding:.7rem .8rem;
           margin-bottom:.5rem; }
.ep-user .name { font-weight:650; }
.ep-user .role { color:var(--ep-ink2); font-size:.8rem; }
.ep-login-brand { text-align:center; margin: 1.2rem 0 1rem 0; }
.ep-login-brand .logo { font-size:2.6rem; }
.ep-login-brand h1 { font-size:2rem; margin:.2rem 0 0 0; }
.ep-login-brand p { color:var(--ep-ink2); margin:.2rem 0 0 0; }
div[data-testid="stMetric"] { background:#fff; border:1px solid var(--ep-line); border-radius:12px;
                              padding:.6rem .9rem; }
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def _register_template() -> None:
    template = go.layout.Template()
    template.layout = go.Layout(
        font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif', size=12.5, color=INK_2),
        title=dict(font=dict(size=14.5, color=INK), x=0, xanchor="left", y=0.97),
        colorway=SERIES,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=SURFACE,
        margin=dict(l=10, r=10, t=48, b=10),
        xaxis=dict(gridcolor=GRID, linecolor=AXIS, zeroline=False, ticks="", title_font=dict(size=12)),
        yaxis=dict(gridcolor=GRID, linecolor=AXIS, zeroline=False, ticks="", title_font=dict(size=12)),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1, title_text="",
                    font=dict(size=11.5)),
        hoverlabel=dict(bgcolor="white", bordercolor=AXIS, font=dict(color=INK, size=12)),
        bargap=0.35,
    )
    pio.templates["edupredict"] = template
    pio.templates.default = "edupredict"


_register_template()
