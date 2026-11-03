"""
Assembles everything needed for a student's insight view or PDF report,
and for class-level reports.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from core.scope import AccessDenied, Scope
from reports.pdf_report import build_class_report, build_student_report
from services import analytics_service, recommendation_service
from services.academic_service import list_records
from services.student_service import get_student


@dataclass
class StudentInsight:
    student: dict
    records: pd.DataFrame
    risk_row: dict | None
    assessment: pd.DataFrame
    recommendations: list
    scenarios: pd.DataFrame
    study_plan: pd.DataFrame
    past_difficulties: pd.DataFrame


def student_insight(student_id: str, scope: Scope) -> StudentInsight:
    student = get_student(student_id, scope)
    if student is None:
        raise AccessDenied("Student not found or not accessible.")
    records = list_records(scope, student_id=student_id)
    risk = analytics_service.student_risk_table(scope, records)
    risk_row = risk.iloc[0].to_dict() if not risk.empty else None
    assessment = recommendation_service.subject_assessment(records)
    recs = recommendation_service.generate_recommendations(records, risk_row["overall_risk"] if risk_row else None)
    scenarios = recommendation_service.improvement_scenarios(assessment, records)
    plan = recommendation_service.study_plan(assessment)
    past = recommendation_service.past_difficulties(records)
    return StudentInsight(student, records, risk_row, assessment, recs, scenarios, plan, past)


def student_report_pdf(student_id: str, scope: Scope, generated_by: str) -> bytes:
    ins = student_insight(student_id, scope)
    return build_student_report(ins.student, ins.records, ins.risk_row, ins.assessment, ins.recommendations,
                                ins.scenarios, ins.study_plan, generated_by)


def class_report_pdf(scope: Scope, generated_by: str, department: str | None = None,
                     semester: int | None = None, section: str | None = None) -> bytes:
    df = analytics_service.records_frame(scope, department, semester, section)
    risk = analytics_service.student_risk_table(scope, df)
    kpis = analytics_service.kpis(scope, df, risk)
    current = analytics_service.current_records(df)
    subj = analytics_service.subject_averages(current)
    filters = {
        "Department": department or "All",
        "Semester": semester or "All",
        "Section": section or "All",
        "Students": len(risk),
    }
    title = "Class Performance Report" if (department and semester and section) else "Cohort Performance Report"
    return build_class_report(title, filters, df, risk, kpis, subj, generated_by)
