"""
Plotly figure builders used across the dashboards.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import config
from ui.theme import AXIS, GRID, INK, INK_2, MUTED, RISK, SERIES


def _empty(title: str, message: str = "No data for the current selection") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font=dict(color=MUTED, size=13), x=0.5, y=0.5,
                       xref="paper", yref="paper")
    fig.update_layout(title=title, height=300, xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig


def _risk_lines(fig: go.Figure, axis: str = "x") -> None:
    """Dotted reference lines at the risk cut-offs, labelled inside the plot so they never hit the legend."""
    labels = [(config.MODERATE_RISK_MIN, "High risk < 50", "right"), (config.LOW_RISK_MIN, "Low risk ≥ 70", "left")]
    for value, label, anchor in labels:
        if axis == "x":
            fig.add_vline(x=value, line_dash="dot", line_color=MUTED, line_width=1)
            fig.add_annotation(x=value, y=1, xref="x", yref="paper", yanchor="top", xanchor=anchor,
                               xshift=-4 if anchor == "right" else 4, text=label, showarrow=False,
                               font=dict(size=10.5, color=INK_2), bgcolor="rgba(252,252,251,0.85)")
        else:
            fig.add_hline(y=value, line_dash="dot", line_color=MUTED, line_width=1)


def performance_distribution(df: pd.DataFrame) -> go.Figure:
    title = "Student performance distribution"
    if df.empty:
        return _empty(title)
    fig = go.Figure()
    actual = df["final_marks"].dropna()
    pending = df[df["final_marks"].isna()]["predicted_marks"].dropna()
    bins = dict(start=0, end=100, size=5)
    if len(actual):
        fig.add_trace(go.Histogram(x=actual, xbins=bins, name="Actual final marks (completed)",
                                   marker_color=SERIES[0], opacity=0.85,
                                   hovertemplate="Marks %{x}<br>%{y} records<extra>Actual</extra>"))
    if len(pending):
        fig.add_trace(go.Histogram(x=pending, xbins=bins, name="Predicted marks (upcoming exam)",
                                   marker_color=SERIES[1], opacity=0.85,
                                   hovertemplate="Marks %{x}<br>%{y} records<extra>Predicted</extra>"))
    fig.update_layout(title=title, barmode="group", height=360, bargap=0.08,
                      xaxis_title="Marks out of 100", yaxis_title="Subject records")
    _risk_lines(fig)
    return fig


def subject_averages(subj: pd.DataFrame, top: int = 15) -> go.Figure:
    title = "Subject-wise average marks"
    if subj.empty:
        return _empty(title)
    data = subj.dropna(subset=["avg_marks"]).sort_values("avg_marks")
    if len(data) > top:
        data = pd.concat([data.head(top // 2 + 1), data.tail(top // 2)]).drop_duplicates("subject")
        title += f" (lowest and highest {top})"
    fig = go.Figure(go.Bar(
        y=data["subject"], x=data["avg_marks"], orientation="h", marker_color=SERIES[0],
        text=[f"{v:.1f}" for v in data["avg_marks"]], textposition="outside", cliponaxis=False,
        customdata=np.stack([data["avg_internal_pct"], data["avg_attendance"], data["records"]], axis=1),
        hovertemplate="<b>%{y}</b><br>Average marks %{x:.1f}<br>Internal %{customdata[0]:.1f}%"
                      "<br>Attendance %{customdata[1]:.1f}%<br>%{customdata[2]} records<extra></extra>",
    ))
    fig.update_layout(title=title, height=max(320, 26 * len(data) + 110), xaxis=dict(range=[0, 108],
                      title="Average marks (actual, or predicted if pending)"), yaxis_title="")
    _risk_lines(fig)
    return fig


def attendance_distribution(df: pd.DataFrame) -> go.Figure:
    title = "Attendance distribution"
    if df.empty:
        return _empty(title)
    fig = go.Figure(go.Histogram(x=df["attendance"], xbins=dict(start=30, end=100, size=2.5),
                                 marker_color=SERIES[2],
                                 hovertemplate="Attendance %{x}%<br>%{y} records<extra></extra>"))
    fig.add_vline(x=config.ATTENDANCE_MINIMUM, line_dash="dot", line_color=RISK[config.RISK_HIGH], line_width=1.5,
                  annotation_text=f"{config.ATTENDANCE_MINIMUM:.0f}% minimum", annotation_position="top left",
                  annotation_font=dict(size=10.5, color=INK_2))
    below = (df["attendance"] < config.ATTENDANCE_MINIMUM).mean() * 100
    fig.update_layout(title=f"{title} · {below:.0f}% below {config.ATTENDANCE_MINIMUM:.0f}%", height=340, bargap=0.06,
                      xaxis_title="Attendance (%)", yaxis_title="Subject records")
    return fig


def semester_performance(sem: pd.DataFrame) -> go.Figure:
    title = "Semester-wise performance"
    if sem.empty:
        return _empty(title)
    fig = go.Figure()
    labels = [f"Sem {s}" for s in sem["semester"]]
    fig.add_trace(go.Bar(x=labels, y=sem["avg_final"], name="Average actual final marks", marker_color=SERIES[0],
                         text=[f"{v:.1f}" if pd.notna(v) else "" for v in sem["avg_final"]], textposition="outside",
                         hovertemplate="%{x}<br>Actual %{y:.1f}<extra></extra>"))
    fig.add_trace(go.Bar(x=labels, y=sem["avg_predicted"], name="Average predicted marks", marker_color=SERIES[1],
                         text=[f"{v:.1f}" if pd.notna(v) else "" for v in sem["avg_predicted"]],
                         textposition="outside", hovertemplate="%{x}<br>Predicted %{y:.1f}<extra></extra>"))
    fig.update_layout(title=title, barmode="group", height=360, yaxis=dict(range=[0, 100], title="Marks"),
                      bargap=0.3, bargroupgap=0.08)
    return fig


def risk_distribution(risk: pd.DataFrame, title: str = "Students by overall risk level") -> go.Figure:
    if risk.empty:
        return _empty(title)
    counts = risk["overall_risk"].value_counts()
    levels = [config.RISK_HIGH, config.RISK_MODERATE, config.RISK_LOW]
    values = [int(counts.get(lvl, 0)) for lvl in levels]
    total = max(sum(values), 1)
    fig = go.Figure(go.Bar(
        y=levels[::-1], x=values[::-1], orientation="h", marker_color=[RISK[lvl] for lvl in levels[::-1]],
        text=[f"{v}  ({v / total:.0%})" for v in values[::-1]], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{x} students<extra></extra>",
    ))
    fig.update_layout(title=title, height=280, xaxis=dict(range=[0, max(values + [1]) * 1.3], title="Students"),
                      yaxis_title="", bargap=0.4)
    return fig


def internal_vs_final(df: pd.DataFrame) -> go.Figure:
    title = "Internal test average vs final examination marks"
    done = df.dropna(subset=["final_marks"]) if not df.empty else df
    if done.empty:
        return _empty(title, "No completed records with final marks yet")
    fig = go.Figure(go.Scattergl(
        x=done["internal_avg_pct"], y=done["final_marks"], mode="markers", name="Completed subject records",
        marker=dict(size=7, color=SERIES[0], opacity=0.45, line=dict(width=0.5, color="white")),
        customdata=np.stack([done["full_name"], done["subject"], done["attendance"]], axis=1),
        hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}<br>Internal %{x:.0f}% · Final %{y:.0f}"
                      "<br>Attendance %{customdata[2]:.0f}%<extra></extra>",
    ))
    if len(done) > 2:
        slope, intercept = np.polyfit(done["internal_avg_pct"], done["final_marks"], 1)
        xs = np.array([done["internal_avg_pct"].min(), done["internal_avg_pct"].max()])
        r = np.corrcoef(done["internal_avg_pct"], done["final_marks"])[0, 1]
        fig.add_trace(go.Scatter(x=xs, y=slope * xs + intercept, mode="lines", name=f"Trend (r = {r:.2f})",
                                 line=dict(color=SERIES[1], width=2), hoverinfo="skip"))
    fig.update_layout(title=title, height=380, xaxis=dict(title="Internal test average (%)", range=[0, 102]),
                      yaxis=dict(title="Final examination marks", range=[0, 102]))
    return fig


def department_trend(trend: pd.DataFrame) -> go.Figure:
    title = "Academic performance trend by department"
    if trend.empty:
        return _empty(title, "No completed semesters yet")
    fig = go.Figure()
    for i, dept in enumerate(config.DEPARTMENTS):
        d = trend[trend["department"] == dept].sort_values("semester")
        if d.empty:
            continue
        fig.add_trace(go.Scatter(
            x=d["semester"], y=d["avg_final"], mode="lines+markers", name=dept,
            line=dict(color=SERIES[i], width=2), marker=dict(size=8, line=dict(width=2, color="white")),
            hovertemplate=f"<b>{dept}</b><br>Semester %{{x}}<br>Average final %{{y:.1f}}<extra></extra>",
        ))
    fig.update_layout(title=title, height=360, hovermode="x unified",
                      xaxis=dict(title="Semester", dtick=1), yaxis=dict(title="Average final marks", range=[0, 100]))
    return fig


# --------------------------------------------------------------------------- #
# Student-level charts
# --------------------------------------------------------------------------- #
def student_subject_outlook(cur: pd.DataFrame) -> go.Figure:
    title = "Current semester: expected marks by subject"
    if cur.empty:
        return _empty(title)
    data = cur.copy()
    data["marks"] = data["final_marks"].fillna(data["predicted_marks"])
    data = data.dropna(subset=["marks"]).sort_values("marks")
    if data.empty:
        return _empty(title, "No predictions yet")
    kind = np.where(data["final_marks"].notna(), "Actual", "Predicted")
    fig = go.Figure(go.Bar(
        y=data["subject"], x=data["marks"], orientation="h",
        marker_color=[RISK.get(r, SERIES[0]) for r in data["risk_level"].fillna("")],
        text=[f"{m:.0f} · {r}" for m, r in zip(data["marks"], data["risk_level"].fillna("Not assessed"))],
        textposition="outside", cliponaxis=False, customdata=kind,
        hovertemplate="<b>%{y}</b><br>%{customdata} marks %{x:.1f}<extra></extra>",
    ))
    fig.update_layout(title=title, height=max(260, 48 * len(data) + 90),
                      xaxis=dict(range=[0, 125], title="Marks out of 100"), yaxis_title="")
    _risk_lines(fig)
    return fig


def student_history(records: pd.DataFrame) -> go.Figure:
    title = "Semester-wise progress"
    if records.empty:
        return _empty(title)
    g = records.groupby("semester").agg(final=("final_marks", "mean"), predicted=("predicted_marks", "mean"),
                                        internal=("internal_avg_pct", "mean"),
                                        attendance=("attendance", "mean")).reset_index()
    x = [f"Sem {s}" for s in g["semester"]]
    fig = go.Figure()
    series = [("final", "Final exam marks (actual)", SERIES[0]), ("predicted", "Predicted marks", SERIES[1]),
              ("internal", "Internal test %", SERIES[2]), ("attendance", "Attendance %", SERIES[3])]
    for col, name, color in series:
        if g[col].notna().any():
            fig.add_trace(go.Scatter(x=x, y=g[col], name=name, mode="lines+markers",
                                     line=dict(color=color, width=2, dash="dash" if col == "predicted" else "solid"),
                                     marker=dict(size=8, line=dict(width=2, color="white")),
                                     hovertemplate=f"{name}: %{{y:.1f}}<extra></extra>"))
    fig.update_layout(title=title, height=360, hovermode="x unified", yaxis=dict(range=[0, 105], title="Score"))
    return fig


def marks_bullet(predicted: float, reference: float | None = None, reference_label: str = "Actual") -> go.Figure:
    """Horizontal band chart: predicted marks against the High / Moderate / Low risk bands."""
    fig = go.Figure()
    bands = [(0, 50, "#f6d6d6", "High risk"), (50, 70, "#fdeccb", "Moderate"), (70, 100, "#d5efd5", "Low risk")]
    for lo, hi, color, name in bands:
        fig.add_shape(type="rect", x0=lo, x1=hi, y0=0, y1=1, fillcolor=color, line_width=0, layer="below")
        fig.add_annotation(x=(lo + hi) / 2, y=1.0, yanchor="bottom", text=name, showarrow=False,
                           font=dict(size=10.5, color=INK_2))
    fig.add_trace(go.Bar(x=[predicted], y=[0.5], orientation="h", width=0.36, marker_color=INK, name="Predicted",
                         hovertemplate="Predicted %{x:.1f}<extra></extra>"))
    if reference is not None:
        fig.add_trace(go.Scatter(x=[reference, reference], y=[0.12, 0.88], mode="lines", name=reference_label,
                                 line=dict(color=SERIES[0], width=4),
                                 hovertemplate=f"{reference_label} %{{x:.1f}}<extra></extra>"))
    fig.update_layout(height=105, margin=dict(l=6, r=6, t=22, b=6), showlegend=False, plot_bgcolor="rgba(0,0,0,0)",
                      xaxis=dict(range=[0, 100], dtick=10, showgrid=False, linecolor=AXIS),
                      yaxis=dict(range=[0, 1], visible=False), bargap=0)
    return fig


def contributions_chart(contrib: pd.DataFrame, column: str = "marks_effect",
                        title: str = "What drives this prediction") -> go.Figure:
    data = contrib.sort_values(column)
    colors = [SERIES[0] if v >= 0 else SERIES[7] for v in data[column]]
    unit = "marks" if column == "marks_effect" else "risk points"
    fig = go.Figure(go.Bar(
        y=[f"{lbl} ({val:g})" for lbl, val in zip(data["label"], data["value"])], x=data[column], orientation="h",
        marker_color=colors, text=[f"{v:+.1f}" for v in data[column]], textposition="outside", cliponaxis=False,
        customdata=data["typical"],
        hovertemplate="%{y}<br>Typical value %{customdata:g}<br>Effect %{x:+.1f} " + unit + "<extra></extra>",
    ))
    span = max(abs(data[column]).max(), 1) * 1.35
    fig.add_vline(x=0, line_color=AXIS, line_width=1)
    fig.update_layout(title=title, height=320, xaxis=dict(range=[-span, span],
                      title=f"Change in predicted {unit} vs a typical student"), yaxis_title="")
    return fig


def probability_bars(prob_low: float, prob_mod: float, prob_high: float) -> go.Figure:
    levels = [config.RISK_LOW, config.RISK_MODERATE, config.RISK_HIGH]
    values = [prob_low, prob_mod, prob_high]
    fig = go.Figure(go.Bar(
        y=levels[::-1], x=values[::-1], orientation="h", marker_color=[RISK[lvl] for lvl in levels[::-1]],
        text=[f"{v:.0%}" for v in values[::-1]], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{x:.1%}<extra></extra>",
    ))
    fig.update_layout(title="Risk class probabilities (Random Forest)", height=230,
                      xaxis=dict(range=[0, 1.18], tickformat=".0%", title=""), yaxis_title="")
    return fig


# --------------------------------------------------------------------------- #
# Model evaluation charts
# --------------------------------------------------------------------------- #
def actual_vs_predicted(y_true, preds: dict[str, list], title: str = "Actual vs predicted (held-out test set)"):
    fig = go.Figure()
    for i, (name, y_pred) in enumerate(preds.items()):
        fig.add_trace(go.Scattergl(x=y_true, y=y_pred, mode="markers", name=name,
                                   marker=dict(size=7, color=SERIES[i], opacity=0.55, line=dict(width=0.5, color="white")),
                                   hovertemplate="Actual %{x:.1f}<br>Predicted %{y:.1f}<extra>" + name + "</extra>"))
    fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode="lines", name="Perfect prediction",
                             line=dict(color=MUTED, dash="dot", width=1.5), hoverinfo="skip"))
    fig.update_layout(title=title, height=420, xaxis=dict(title="Actual final marks", range=[0, 102]),
                      yaxis=dict(title="Predicted final marks", range=[0, 102]))
    return fig


def residual_histogram(y_true, y_pred, name: str) -> go.Figure:
    resid = np.array(y_pred) - np.array(y_true)
    fig = go.Figure(go.Histogram(x=resid, xbins=dict(size=2), marker_color=SERIES[0],
                                 hovertemplate="Error %{x}<br>%{y} records<extra></extra>"))
    fig.add_vline(x=0, line_color=INK_2, line_width=1)
    fig.update_layout(title=f"Prediction errors · {name}", height=320, bargap=0.06,
                      xaxis_title="Predicted − actual (marks)", yaxis_title="Test records")
    return fig


def feature_importance(importance: dict, title: str, color: str = SERIES[0]) -> go.Figure:
    s = pd.Series(importance).rename(index=config.SHORT_LABELS).sort_values()
    fig = go.Figure(go.Bar(y=s.index, x=s.values, orientation="h", marker_color=color,
                           text=[f"{v:.3f}" for v in s.values], textposition="outside", cliponaxis=False,
                           hovertemplate="%{y}: %{x:.4f}<extra></extra>"))
    fig.update_layout(title=title, height=300, xaxis=dict(range=[min(0, s.min() * 1.3), max(s.max(), 0) * 1.3]),
                      yaxis_title="")
    return fig


def confusion_matrix_chart(matrix: list[list[int]], labels: list[str], title: str) -> go.Figure:
    z = np.array(matrix)
    totals = z.sum(axis=1, keepdims=True)
    pct = np.divide(z, totals, out=np.zeros_like(z, dtype=float), where=totals != 0)
    text = [[f"{z[i][j]}<br>({pct[i][j]:.0%})" for j in range(len(labels))] for i in range(len(labels))]
    fig = go.Figure(go.Heatmap(
        z=pct, x=labels, y=labels, customdata=text, colorscale=[[0, "#f4f8fe"], [1, "#184f95"]],
        showscale=False, hovertemplate="Actual %{y}<br>Predicted %{x}<br>%{customdata}<extra></extra>",
        zmin=0, zmax=1, xgap=2, ygap=2,
    ))
    for i in range(len(labels)):
        for j in range(len(labels)):
            fig.add_annotation(x=labels[j], y=labels[i], text=text[i][j], showarrow=False,
                               font=dict(color="white" if pct[i][j] > 0.5 else INK, size=12.5))
    fig.update_layout(title=title, height=360, xaxis=dict(title="Predicted", side="bottom", gridcolor=GRID),
                      yaxis=dict(title="Actual", autorange="reversed"))
    return fig


def metric_comparison(names: list[str], metrics: dict[str, list[float]], title: str) -> go.Figure:
    fig = go.Figure()
    for i, (name, values) in enumerate(zip(names, zip(*metrics.values()))):
        fig.add_trace(go.Bar(x=list(metrics.keys()), y=list(values), name=name, marker_color=SERIES[i],
                             text=[f"{v:.3f}" for v in values], textposition="outside", cliponaxis=False,
                             hovertemplate=f"{name}<br>%{{x}}: %{{y:.3f}}<extra></extra>"))
    fig.update_layout(title=title, barmode="group", height=340, bargap=0.35, bargroupgap=0.08)
    return fig
