"""Analytics, recommendations and PDF reports."""
import config
from core import database as db
from services import analytics_service as an
from services import recommendation_service as rs
from services import report_service
from services.academic_service import list_records


def test_overall_risk_rule():
    assert an.overall_risk(45, 0) == config.RISK_HIGH
    assert an.overall_risk(65, 2) == config.RISK_HIGH
    assert an.overall_risk(65, 0) == config.RISK_MODERATE
    assert an.overall_risk(75, 1) == config.RISK_MODERATE
    assert an.overall_risk(75, 0) == config.RISK_LOW


def test_kpis_match_database(admin_scope):
    df = list_records(admin_scope)
    risk = an.student_risk_table(admin_scope, df)
    k = an.kpis(admin_scope, df, risk)
    assert k["students"] == db.table_count("students")
    assert k["records"] == db.table_count("academic_records")
    assert k["high_risk"] + k["moderate_risk"] + k["low_risk"] == len(risk)
    assert abs(k["avg_attendance"] - db.query_scalar("SELECT AVG(attendance) FROM academic_records")) < 1e-6


def test_teacher_dashboard_is_scoped(teacher_scope):
    risk = an.student_risk_table(teacher_scope)
    assert len(risk) == 24 and set(risk["department"]) == {"Computer Science (AIML)"}


def test_demo_students_cover_three_risk_levels():
    from core.scope import Scope
    from services.auth_service import get_user_by_username
    levels = []
    for name in ("student1", "student2", "student3"):
        sc = Scope.from_user(get_user_by_username(name))
        levels.append(an.student_risk_table(sc).iloc[0]["overall_risk"])
    assert levels == [config.RISK_MODERATE, config.RISK_HIGH, config.RISK_LOW]


def test_weak_subjects_and_recommendations(admin_scope):
    risk = an.student_risk_table(admin_scope)
    high = risk[risk["overall_risk"] == config.RISK_HIGH].iloc[0]
    records = list_records(admin_scope, student_id=high["student_id"])
    weak = rs.weak_subjects(records)
    assert not weak.empty
    recs = rs.generate_recommendations(records, config.RISK_HIGH)
    assert recs[0].priority == "High" and any(r.category == "Academic Support" for r in recs)
    plan = rs.study_plan(rs.subject_assessment(records))
    assert (plan["recommended_hours"] >= plan["current_hours"]).all()
    assert rs.teacher_actions(high.to_dict())


def test_improvement_scenarios_increase_marks(admin_scope):
    risk = an.student_risk_table(admin_scope)
    sid = risk[risk["overall_risk"] == config.RISK_MODERATE].iloc[0]["student_id"]
    records = list_records(admin_scope, student_id=sid)
    scen = rs.improvement_scenarios(rs.subject_assessment(records), records)
    assert not scen.empty and (scen["gain"] >= -0.5).all()


def test_student_and_class_pdf(admin_scope, teacher_scope):
    sid = list_records(teacher_scope).iloc[0]["student_id"]
    pdf = report_service.student_report_pdf(sid, teacher_scope, "Test Teacher")
    assert pdf[:4] == b"%PDF" and len(pdf) > 20_000
    cls = report_service.class_report_pdf(teacher_scope, "Test Teacher", "Computer Science (AIML)", 5, "A")
    assert cls[:4] == b"%PDF" and len(cls) > 20_000
    cohort = report_service.class_report_pdf(admin_scope, "Admin")
    assert cohort[:4] == b"%PDF"


def test_reset_demo_data_restores_original_state(admin_user, admin_scope):
    from core.seed import reset_demo_data
    from services import student_service

    sid = db.query_scalar("SELECT student_id FROM students LIMIT 1")
    reg = db.query_scalar("SELECT reg_no FROM students WHERE student_id = ?", [sid])
    student_service.delete_student(sid, reg, admin_scope, admin_user)
    assert db.table_count("students") == 143
    summary = reset_demo_data()
    assert summary["seeded"] and db.table_count("students") == 144
    assert db.table_count("users") == 10 and db.table_count("model_registry") == 1
    assert db.query_scalar("SELECT COUNT(*) FROM academic_records r LEFT JOIN latest_predictions p "
                           "ON p.record_id = r.id WHERE p.id IS NULL") == 0
