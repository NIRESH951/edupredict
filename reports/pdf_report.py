"""
PDF report generation with ReportLab (charts drawn with Matplotlib).

  * build_student_report  - individual academic performance report
  * build_class_report    - class / cohort performance report
Both return the PDF as bytes, ready for st.download_button.
"""
from __future__ import annotations

import io
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_CENTER  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import mm  # noqa: E402
from reportlab.platypus import (  # noqa: E402
    Image,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

import config  # noqa: E402

PRIMARY = colors.HexColor("#1f3a8a")
INK = colors.HexColor("#0b0b0b")
MUTED = colors.HexColor("#52514e")
GRID = colors.HexColor("#e1e0d9")
ZEBRA = colors.HexColor("#f6f6f3")
RISK_HEX = config.RISK_COLORS
SERIES_BLUE = "#2a78d6"

_styles = getSampleStyleSheet()
STYLES = {
    "title": ParagraphStyle("t", parent=_styles["Title"], fontSize=18, textColor=PRIMARY, spaceAfter=2),
    "subtitle": ParagraphStyle("st", parent=_styles["Normal"], fontSize=9.5, textColor=MUTED,
                               alignment=TA_CENTER, spaceAfter=8),
    "h2": ParagraphStyle("h2", parent=_styles["Heading2"], fontSize=12.5, textColor=PRIMARY,
                         spaceBefore=10, spaceAfter=5),
    "h3": ParagraphStyle("h3", parent=_styles["Heading3"], fontSize=10.5, textColor=INK,
                         spaceBefore=6, spaceAfter=2),
    "body": ParagraphStyle("b", parent=_styles["Normal"], fontSize=9, leading=12.5, textColor=INK),
    "small": ParagraphStyle("s", parent=_styles["Normal"], fontSize=7.8, leading=10, textColor=MUTED),
    "cell": ParagraphStyle("c", parent=_styles["Normal"], fontSize=8, leading=10, textColor=INK),
    "head": ParagraphStyle("hd", parent=_styles["Normal"], fontSize=7.2, leading=8.6, textColor=colors.white,
                           fontName="Helvetica-Bold"),
    "bullet": ParagraphStyle("bl", parent=_styles["Normal"], fontSize=8.8, leading=12, leftIndent=10,
                             bulletIndent=2, textColor=INK),
}


def _clean(text) -> str:
    """Replace glyphs missing from the base-14 PDF fonts (plain table cells / canvas text)."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return "-"
    s = str(text)
    for a, b in {"→": "->", "≥": ">=", "≤": "<=", "–": "-", "—": "-", "•": "-"}.items():
        s = s.replace(a, b)
    return s.encode("latin-1", "ignore").decode("latin-1")


def _safe(text) -> str:
    """_clean + XML escaping, for use inside Paragraph markup."""
    return _clean(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _fmt(value, digits: int = 1, suffix: str = "") -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "-"
    try:
        return f"{float(value):.{digits}f}{suffix}"
    except (TypeError, ValueError):
        return str(value)


def _table(data: list[list], col_widths: list[float], header_bg=PRIMARY, risk_col: int | None = None,
           font_size: float = 8) -> Table:
    header = [Paragraph(_safe(c), STYLES["head"]) for c in data[0]]
    body = [[(Paragraph(_safe(c), STYLES["cell"]) if len(c) > 28 else _clean(c)) if isinstance(c, str) else c
             for c in row] for row in data[1:]]
    wrapped = [header] + body
    t = Table(wrapped, colWidths=col_widths, repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), header_bg),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.4, GRID),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    for i in range(1, len(data)):
        if i % 2 == 0:
            style.append(("BACKGROUND", (0, i), (-1, i), ZEBRA))
        if risk_col is not None:
            level = str(data[i][risk_col])
            if level in RISK_HEX:
                style.append(("TEXTCOLOR", (risk_col, i), (risk_col, i),
                              colors.HexColor("#b26a00" if level == config.RISK_MODERATE else RISK_HEX[level])))
                style.append(("FONTNAME", (risk_col, i), (risk_col, i), "Helvetica-Bold"))
    t.setStyle(TableStyle(style))
    return t


def _kpi_strip(items: list[tuple[str, str]], width: float) -> Table:
    data = [[Paragraph(f"<font size=7.5 color='#52514e'>{_safe(k)}</font><br/>"
                       f"<font size=13><b>{_safe(v)}</b></font>", STYLES["body"]) for k, v in items]]
    t = Table(data, colWidths=[width / len(items)] * len(items))
    t.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.6, GRID),
        ("INNERGRID", (0, 0), (-1, -1), 0.6, GRID),
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8f9fc")),
        ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
    ]))
    return t


def _fig_to_image(fig, width_mm: float) -> Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    img = Image(buf)
    ratio = img.imageHeight / img.imageWidth
    img.drawWidth = width_mm * mm
    img.drawHeight = width_mm * mm * ratio
    return img


def _style_axes(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors="#52514e", labelsize=7.5)
    ax.grid(axis="x", color="#e1e0d9", linewidth=0.6)
    ax.set_axisbelow(True)


def _header_footer(title: str, generated_by: str):
    stamp = datetime.now().strftime("%d %b %Y, %I:%M %p")

    def draw(canvas, doc):
        canvas.saveState()
        w, h = A4
        canvas.setFillColor(PRIMARY)
        canvas.rect(0, h - 14 * mm, w, 14 * mm, stroke=0, fill=1)
        canvas.setFillColor(colors.white)
        canvas.setFont("Helvetica-Bold", 10.5)
        canvas.drawString(15 * mm, h - 9 * mm, f"{config.APP_NAME}  |  {config.INSTITUTION_NAME}")
        canvas.setFont("Helvetica", 8.5)
        canvas.drawRightString(w - 15 * mm, h - 9 * mm, _clean(title))
        canvas.setFillColor(MUTED)
        canvas.setFont("Helvetica", 7.5)
        canvas.drawString(15 * mm, 9 * mm,
                          f"Generated {stamp} by {_clean(generated_by)}  |  AI-assisted decision support - "
                          "review with a teacher before acting on it.")
        canvas.drawRightString(w - 15 * mm, 9 * mm, f"Page {doc.page}")
        canvas.restoreState()

    return draw


def _doc(buffer) -> SimpleDocTemplate:
    return SimpleDocTemplate(buffer, pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm,
                             topMargin=20 * mm, bottomMargin=16 * mm,
                             title=f"{config.APP_NAME} report", author=config.APP_NAME)


# --------------------------------------------------------------------------- #
# Student report
# --------------------------------------------------------------------------- #
def _student_subject_chart(cur: pd.DataFrame):
    data = cur.copy()
    data["marks"] = data["final_marks"].fillna(data["predicted_marks"])
    data = data.dropna(subset=["marks"]).sort_values("marks")
    if data.empty:
        return None
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(data) + 0.9))
    bar_colors = [RISK_HEX.get(r, SERIES_BLUE) for r in data["risk_level"].fillna("")]
    ax.barh(data["subject"], data["marks"], color=bar_colors, height=0.55)
    for y, (m, r) in enumerate(zip(data["marks"], data["risk_level"].fillna(""))):
        ax.text(m + 1, y, f"{m:.0f}  {r}", va="center", fontsize=7, color="#0b0b0b")
    ax.axvline(config.MODERATE_RISK_MIN, color="#898781", linestyle="--", linewidth=0.8)
    ax.axvline(config.LOW_RISK_MIN, color="#898781", linestyle=":", linewidth=0.8)
    ax.set_xlim(0, 118)
    ax.set_xlabel("Marks out of 100 (predicted, or actual where available)", fontsize=7.5, color="#52514e")
    _style_axes(ax)
    return fig


def _student_history_chart(records: pd.DataFrame):
    done = records.dropna(subset=["final_marks"])
    if done.empty:
        return None
    sem = done.groupby("semester").agg(final=("final_marks", "mean"), internal=("internal_avg_pct", "mean"),
                                       attendance=("attendance", "mean")).reset_index()
    fig, ax = plt.subplots(figsize=(7.2, 2.3))
    ax.plot(sem["semester"], sem["final"], marker="o", color=SERIES_BLUE, linewidth=2, label="Final exam marks")
    ax.plot(sem["semester"], sem["internal"], marker="s", color="#eb6834", linewidth=2, label="Internal test %")
    for x, y in zip(sem["semester"], sem["final"]):
        ax.annotate(f"{y:.0f}", (x, y), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7)
    ax.set_xticks(sem["semester"])
    ax.set_xticklabels([f"Sem {s}" for s in sem["semester"]])
    ax.set_ylim(0, 105)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    _style_axes(ax)
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.6)
    ax.grid(axis="x", visible=False)
    return fig


def build_student_report(student: dict, records: pd.DataFrame, risk_row: dict | None,
                         assessment: pd.DataFrame, recommendations: list, scenarios: pd.DataFrame,
                         study_plan: pd.DataFrame, generated_by: str) -> bytes:
    from services.analytics_service import current_records

    buffer = io.BytesIO()
    doc = _doc(buffer)
    width = doc.width
    story = [
        Paragraph("Student Academic Performance Report", STYLES["title"]),
        Paragraph(_safe(f"{student['full_name']}  |  {student['reg_no']}  |  {student['department']}"),
                  STYLES["subtitle"]),
    ]

    profile = [
        ["Student ID", student["student_id"], "Registration No.", student["reg_no"]],
        ["Department", student["department"], "Semester / Section", f"{student['semester']} / {student['section']}"],
        ["Academic Year", student["academic_year"], "Enrolled", student["enrollment_date"]],
        ["Email", student.get("email") or "-", "Overall Risk", (risk_row or {}).get("overall_risk", "-")],
    ]
    t = Table([[Paragraph(_safe(c), STYLES["cell"]) for c in row] for row in profile],
              colWidths=[28 * mm, 62 * mm, 32 * mm, width - 122 * mm])
    t.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"), ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
        ("TEXTCOLOR", (0, 0), (0, -1), MUTED), ("TEXTCOLOR", (2, 0), (2, -1), MUTED),
        ("GRID", (0, 0), (-1, -1), 0.4, GRID), ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fbfbfa")),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story += [t, Spacer(1, 6)]

    cur = current_records(records)
    if risk_row:
        story.append(_kpi_strip([
            ("Avg predicted marks", _fmt(risk_row.get("avg_predicted"))),
            ("Avg attendance", _fmt(risk_row.get("avg_attendance"), suffix="%")),
            ("Avg internal score", _fmt(risk_row.get("avg_internal_pct"), suffix="%")),
            ("High-risk subjects", str(risk_row.get("high_risk_subjects", 0))),
            ("Overall risk", risk_row.get("overall_risk", "-")),
        ], width))

    story.append(Paragraph(f"Current Semester Outlook (Semester {int(cur['semester'].max()) if not cur.empty else '-'})",
                           STYLES["h2"]))
    if cur.empty:
        story.append(Paragraph("No academic records are available for the current semester.", STYLES["body"]))
    else:
        rows = [["Subject", "Attend.", "Test 1", "Test 2", "Asgn.", "Hrs", "Pred.", "Actual", "Category",
                 "Risk"]]
        for r in cur.itertuples():
            rows.append([r.subject, _fmt(r.attendance, 0, "%"), _fmt(r.internal1, 1), _fmt(r.internal2, 1),
                         _fmt(r.assignment, 1), _fmt(r.study_hours, 1), _fmt(r.predicted_marks),
                         _fmt(r.final_marks), r.performance_category or "-", r.risk_level or "-"])
        story.append(_table(rows, [40 * mm, 14 * mm, 12 * mm, 12 * mm, 13 * mm, 13 * mm, 16 * mm, 13 * mm,
                                   23 * mm, width - 156 * mm], risk_col=9, font_size=7.5))
        fig = _student_subject_chart(cur)
        if fig is not None:
            story += [Spacer(1, 6), _fig_to_image(fig, 175),
                      Paragraph("Dashed line: 50 marks (High Risk below). Dotted line: 70 marks (Low Risk at or above).",
                                STYLES["small"])]

    if not assessment.empty:
        story.append(Paragraph("Subject Strength Analysis", STYLES["h2"]))
        rows = [["Subject", "Status", "Marks", "Reasons"]]
        for r in assessment.itertuples():
            rows.append([r.subject, r.level, _fmt(r.marks), "; ".join(r.issues) if r.issues else "No concerns"])
        story.append(_table(rows, [48 * mm, 24 * mm, 15 * mm, width - 87 * mm], font_size=7.5))

    if scenarios is not None and not scenarios.empty:
        story.append(KeepTogether([
            Paragraph("What Could Change Before the Exam (model estimate)", STYLES["h2"]),
            _table([["Subject", "Now", "If improved", "Gain", "Changes assumed"]] +
                   [[r.subject, _fmt(r.current_prediction), _fmt(r.improved_prediction), f"{r.gain:+.1f}",
                     r.changes] for r in scenarios.itertuples()],
                   [45 * mm, 14 * mm, 20 * mm, 14 * mm, width - 93 * mm], font_size=7.5),
            Paragraph("Estimates from the Random Forest model; internal test marks are kept as they are. "
                      "These are not guarantees.", STYLES["small"]),
        ]))

    story.append(Paragraph("Personalised Study Recommendations", STYLES["h2"]))
    for rec in recommendations:
        rec = rec if isinstance(rec, dict) else rec.as_dict()
        block = [Paragraph(f"<b>[{_safe(rec['priority'])} priority] {_safe(rec['category'])}:</b> "
                           f"{_safe(rec['title'])}", STYLES["h3"]),
                 Paragraph(_safe(rec["detail"]), STYLES["body"])]
        block += [Paragraph(_safe(a), STYLES["bullet"], bulletText="-") for a in rec["actions"]]
        story.append(KeepTogether(block))

    if study_plan is not None and not study_plan.empty:
        story.append(KeepTogether([
            Paragraph("Weekly Study Plan", STYLES["h2"]),
            _table([["Subject", "Status", "Now (hrs)", "Target (hrs)", "Focus"]] +
                   [[r.subject, r.status, _fmt(r.current_hours), _fmt(r.recommended_hours), r.focus]
                    for r in study_plan.itertuples()],
                   [46 * mm, 26 * mm, 18 * mm, 20 * mm, width - 110 * mm], font_size=7.5),
            Paragraph(f"Total: {study_plan['current_hours'].sum():.1f} -> "
                      f"{study_plan['recommended_hours'].sum():.1f} hours per week.", STYLES["small"]),
        ]))

    fig = _student_history_chart(records)
    if fig is not None:
        story.append(Paragraph("Academic History", STYLES["h2"]))
        story.append(_fig_to_image(fig, 175))
        done = records.dropna(subset=["final_marks"]).sort_values(["semester", "subject"])
        rows = [["Sem", "Subject", "Attend.", "Internal %", "Assign.", "Final", "Predicted"]]
        for r in done.itertuples():
            rows.append([str(r.semester), r.subject, _fmt(r.attendance, 0, "%"), _fmt(r.internal_avg_pct, 0, "%"),
                         _fmt(r.assignment, 1), _fmt(r.final_marks), _fmt(r.predicted_marks)])
        story.append(_table(rows, [10 * mm, 66 * mm, 18 * mm, 20 * mm, 17 * mm, 18 * mm, width - 149 * mm],
                            font_size=7.5))

    story += [Spacer(1, 10), Paragraph(
        "About this report: predictions come from machine-learning models trained on past academic records "
        "(attendance, internal tests, assignments, study hours and previous GPA). They estimate likely outcomes "
        "to help plan support; they are not final results and do not replace the judgement of teachers.",
        STYLES["small"])]

    doc.build(story, onFirstPage=_header_footer("Student Report", generated_by),
              onLaterPages=_header_footer("Student Report", generated_by))
    return buffer.getvalue()


# --------------------------------------------------------------------------- #
# Class report
# --------------------------------------------------------------------------- #
def _risk_chart(risk: pd.DataFrame):
    counts = risk["overall_risk"].value_counts()
    levels = [lvl for lvl in config.RISK_LEVELS]
    values = [int(counts.get(lvl, 0)) for lvl in levels]
    fig, ax = plt.subplots(figsize=(3.4, 2.0))
    ax.barh(levels[::-1], values[::-1], color=[RISK_HEX[lvl] for lvl in levels[::-1]], height=0.55)
    for y, v in enumerate(values[::-1]):
        ax.text(v + 0.2, y, str(v), va="center", fontsize=7.5)
    ax.set_xlim(0, max(values + [1]) * 1.25)
    ax.set_title("Students by overall risk", fontsize=8, color="#0b0b0b", loc="left")
    _style_axes(ax)
    return fig


def _subject_chart(subj: pd.DataFrame):
    data = subj.sort_values("avg_marks").tail(12)
    fig, ax = plt.subplots(figsize=(3.8, 0.28 * len(data) + 0.8))
    ax.barh(data["subject"], data["avg_marks"], color=SERIES_BLUE, height=0.55)
    for y, v in enumerate(data["avg_marks"]):
        ax.text(v + 1, y, f"{v:.0f}", va="center", fontsize=6.5)
    ax.set_xlim(0, 110)
    ax.tick_params(axis="y", labelsize=6.5)
    ax.set_title("Average marks by subject", fontsize=8, color="#0b0b0b", loc="left")
    _style_axes(ax)
    return fig


def build_class_report(title: str, filters: dict, records: pd.DataFrame, risk: pd.DataFrame, kpis: dict,
                       subject_avgs: pd.DataFrame, generated_by: str) -> bytes:
    buffer = io.BytesIO()
    doc = _doc(buffer)
    width = doc.width
    filter_text = "  |  ".join(f"{k}: {v}" for k, v in filters.items())
    story = [Paragraph(_safe(title), STYLES["title"]), Paragraph(_safe(filter_text), STYLES["subtitle"])]

    story.append(_kpi_strip([
        ("Students", str(len(risk))),
        ("Avg attendance", _fmt(kpis.get("avg_attendance"), suffix="%")),
        ("Avg internal", _fmt(kpis.get("avg_internal_pct"), suffix="%")),
        ("Avg predicted", _fmt(kpis.get("avg_predicted"))),
        ("High risk", str(kpis.get("high_risk", 0))),
        ("Moderate risk", str(kpis.get("moderate_risk", 0))),
    ], width))
    story.append(Spacer(1, 8))

    charts = []
    if not risk.empty:
        charts.append(_fig_to_image(_risk_chart(risk), 80))
    if subject_avgs is not None and not subject_avgs.empty:
        charts.append(_fig_to_image(_subject_chart(subject_avgs), 95))
    if charts:
        ct = Table([charts], colWidths=[width * 0.45, width * 0.55])
        ct.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
        story.append(ct)

    at_risk = risk[risk["overall_risk"].isin([config.RISK_HIGH, config.RISK_MODERATE])]
    story.append(Paragraph(f"Students Needing Attention ({len(at_risk)})", STYLES["h2"]))
    if at_risk.empty:
        story.append(Paragraph("No students are currently at High or Moderate risk.", STYLES["body"]))
    else:
        rows = [["Reg. No.", "Name", "Cls", "Attend.", "Internal", "Pred.", "High-risk subjects",
                 "Weakest subject", "Risk"]]
        for r in at_risk.itertuples():
            rows.append([r.reg_no, r.full_name, f"S{r.current_semester}{r.section}", _fmt(r.avg_attendance, 0, "%"),
                         _fmt(r.avg_internal_pct, 0, "%"), _fmt(r.avg_predicted), str(r.high_risk_subjects),
                         r.weakest_subject or "-", r.overall_risk])
        story.append(_table(rows, [22 * mm, 30 * mm, 11 * mm, 14 * mm, 14 * mm, 15 * mm, 16 * mm, 38 * mm,
                                   width - 160 * mm], risk_col=8, font_size=7.2))

    top = risk[risk["overall_risk"] == config.RISK_LOW].sort_values("avg_predicted", ascending=False).head(5)
    if not top.empty:
        story.append(Paragraph("Top Performers (by predicted marks)", STYLES["h2"]))
        story.append(_table([["Reg. No.", "Name", "Predicted", "Attendance", "Internal"]] +
                            [[r.reg_no, r.full_name, _fmt(r.avg_predicted), _fmt(r.avg_attendance, 0, "%"),
                              _fmt(r.avg_internal_pct, 0, "%")] for r in top.itertuples()],
                            [30 * mm, 50 * mm, 25 * mm, 25 * mm, width - 130 * mm]))

    if subject_avgs is not None and not subject_avgs.empty:
        story.append(Paragraph("Subject-wise Summary", STYLES["h2"]))
        rows = [["Subject", "Records", "Avg final", "Avg predicted", "Avg internal %", "Avg attendance"]]
        for r in subject_avgs.sort_values("subject").itertuples():
            rows.append([r.subject, str(r.records), _fmt(r.avg_final), _fmt(r.avg_predicted),
                         _fmt(r.avg_internal_pct), _fmt(r.avg_attendance)])
        story.append(_table(rows, [62 * mm, 16 * mm, 20 * mm, 24 * mm, 24 * mm, width - 146 * mm]))

    story.append(Paragraph("All Students", STYLES["h2"]))
    rows = [["Reg. No.", "Name", "Class", "Subjects", "Attend.", "Internal", "Predicted", "Risk"]]
    for r in risk.itertuples():
        rows.append([r.reg_no, r.full_name, f"S{r.current_semester}{r.section}", str(r.subjects),
                     _fmt(r.avg_attendance, 0, "%"), _fmt(r.avg_internal_pct, 0, "%"), _fmt(r.avg_predicted),
                     r.overall_risk])
    story.append(_table(rows, [24 * mm, 44 * mm, 14 * mm, 16 * mm, 16 * mm, 16 * mm, 18 * mm, width - 148 * mm],
                        risk_col=7, font_size=7.5))
    story += [Spacer(1, 8), Paragraph(
        "Overall risk rule: High if average predicted marks are below 50 or two or more subjects are predicted "
        "High Risk; Moderate if the average is below 70 or one subject is High Risk; otherwise Low.",
        STYLES["small"])]

    doc.build(story, onFirstPage=_header_footer("Class Report", generated_by),
              onLaterPages=_header_footer("Class Report", generated_by))
    return buffer.getvalue()
