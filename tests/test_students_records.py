"""Modules 3 & 4: student CRUD, academic records, scope rules and CSV import."""
from datetime import date

import pandas as pd
import pytest

from core import database as db
from core.scope import AccessDenied
from core.validation import map_columns, validate_academic_record
from services import academic_service as acs
from services import student_service as ss

NEW_STUDENT = {"reg_no": "1EP24AI900", "full_name": "Test Student", "email": "test.student@example.edu",
               "department": "Computer Science (AIML)", "semester": 5, "academic_year": "2026-27",
               "section": "A", "enrollment_date": date(2024, 8, 1)}

GOOD_RECORD = {"subject": "Machine Learning", "semester": 5, "attendance": 82, "internal1": 30, "internal2": 33,
               "assignment": 15, "study_hours": 6, "previous_gpa": 7.4}


def test_add_update_delete_student(admin_scope, admin_user):
    sid = ss.add_student(NEW_STUDENT, admin_user)
    assert ss.get_student(sid, admin_scope)["reg_no"] == "1EP24AI900"
    with pytest.raises(ss.StudentError):  # duplicate registration number
        ss.add_student({**NEW_STUDENT, "email": "other@example.edu"}, admin_user)
    ss.update_student(sid, {**NEW_STUDENT, "full_name": "Renamed Student"}, admin_scope, admin_user)
    assert ss.get_student(sid, admin_scope)["full_name"] == "Renamed Student"

    rid = acs.add_record({"student_id": sid, **GOOD_RECORD}, admin_scope, admin_user)
    with pytest.raises(ss.StudentError):  # wrong confirmation
        ss.delete_student(sid, "WRONG", admin_scope, admin_user)
    ss.delete_student(sid, "1ep24ai900", admin_scope, admin_user)
    assert ss.get_student(sid, admin_scope) is None
    assert db.query_one("SELECT id FROM academic_records WHERE id = ?", [rid]) is None  # cascade


def test_invalid_student_rejected(admin_user):
    with pytest.raises(ss.StudentError):
        ss.add_student({**NEW_STUDENT, "reg_no": "1EP24AI901", "email": "bad-email", "academic_year": "2026"},
                       admin_user)


def test_record_validation_and_final_marks_optional():
    clean, errors = validate_academic_record({"student_id": "stu0001", **GOOD_RECORD})
    assert errors == [] and clean["final_marks"] is None and clean["student_id"] == "STU0001"
    _, errors = validate_academic_record({"student_id": "STU0001", **GOOD_RECORD, "internal1": 55,
                                          "attendance": "abc"})
    assert len(errors) == 2


def test_record_crud_and_duplicates(admin_scope, admin_user):
    sid = db.query_scalar("SELECT student_id FROM students WHERE semester = 5 LIMIT 1")
    rid = acs.add_record({"student_id": sid, **GOOD_RECORD, "subject": "Test Subject X"}, admin_scope, admin_user)
    with pytest.raises(acs.RecordError):
        acs.add_record({"student_id": sid, **GOOD_RECORD, "subject": "test subject x"}, admin_scope, admin_user)
    with pytest.raises(acs.RecordError):  # semester after the student's current semester
        acs.add_record({"student_id": sid, **GOOD_RECORD, "subject": "Future", "semester": 7}, admin_scope)
    assert acs.update_record(rid, {"attendance": 60}, admin_scope, admin_user) is True
    assert acs.update_record(rid, {"final_marks": 71}, admin_scope, admin_user) is False
    acs.delete_record(rid, admin_scope, admin_user)
    assert acs.get_record(rid, admin_scope) is None


def test_scopes_limit_visibility(admin_scope, teacher_scope, student_scope):
    all_ids = set(ss.list_students(admin_scope)["student_id"])
    teacher_ids = set(ss.list_students(teacher_scope)["student_id"])
    student_ids = set(ss.list_students(student_scope)["student_id"])
    assert teacher_ids < all_ids and len(teacher_ids) == 24  # 3 classes x 8 students
    assert student_ids == {student_scope.student_id}
    other = next(iter(all_ids - teacher_ids))
    with pytest.raises(AccessDenied):
        acs.add_record({"student_id": other, **GOOD_RECORD}, teacher_scope)
    with pytest.raises(AccessDenied):  # students cannot edit records
        acs.add_record({"student_id": student_scope.student_id, **GOOD_RECORD}, student_scope)


def test_column_aliases():
    mapping = map_columns(["Student ID", "Subject", "Sem", "Attendance Percentage", "Internal Test 1",
                           "Internal Test 2", "Assignment Score", "Weekly Study Hours", "Previous GPA", "Final"])
    assert set(mapping.values()) == {"student_id", "subject", "semester", "attendance", "internal1", "internal2",
                                     "assignment", "study_hours", "previous_gpa", "final_marks"}


def test_csv_validation_and_import(teacher_scope, admin_user):
    mine = db.query_df("SELECT s.student_id, s.semester FROM students s JOIN teacher_assignments ta "
                       "ON s.department = ta.department AND s.semester = ta.semester AND s.section = ta.section "
                       "WHERE ta.teacher_id = 'TCH001' LIMIT 2")
    outsider = db.query_scalar("SELECT student_id FROM students WHERE department = 'Information Science' LIMIT 1")
    s1, s2 = mine["student_id"].tolist()
    sem = int(mine["semester"].iloc[0])
    csv = pd.DataFrame([
        [s1, "CSV Subject A", sem, 80, 30, 32, 15, 6, 7.5, ""],          # valid
        [s2, "CSV Subject A", sem, 75, 25, 28, 12, 4, 6.5, 58],          # valid with final marks
        [s1, "CSV Subject A", sem, 80, 30, 32, 15, 6, 7.5, ""],          # duplicate of row 2
        [s1, "CSV Subject B", sem, 120, 30, 32, 15, 6, 7.5, ""],         # attendance out of range
        ["STU9999", "CSV Subject B", sem, 80, 30, 32, 15, 6, 7.5, ""],   # unknown student
        [outsider, "CSV Subject B", 3, 80, 30, 32, 15, 6, 7.5, ""],      # not the teacher's student
        [s1, "", sem, 80, 30, "", 15, 6, 7.5, ""],                       # missing values
    ], columns=["Student ID", "Subject", "Semester", "Attendance %", "Internal Test 1", "Internal Test 2",
                "Assignment", "Study Hours", "Previous GPA", "Final Marks"])
    result = acs.validate_academic_csv(csv.astype(str).replace("", None), teacher_scope)
    assert result.structure_ok and len(result.valid) == 2
    assert result.duplicates_in_file == 1 and result.error_rows == 5
    issues = " ".join(result.errors["issue"])
    assert "does not exist" in issues and "not assigned to you" in issues and "between 0 and 100" in issues
    assert int(result.missing_values.set_index("column").loc["internal2", "missing"]) == 1

    teacher = {"id": None, "username": "teacher1"}
    counts = acs.import_academic_records(result.valid, teacher_scope, "skip", teacher)
    assert counts["inserted"] == 2
    again = acs.validate_academic_csv(csv.iloc[:2].astype(str).replace("", None), teacher_scope)
    assert again.existing_in_db == 2
    assert acs.import_academic_records(again.valid, teacher_scope, "skip", teacher)["skipped"] == 2
    assert acs.import_academic_records(again.valid, teacher_scope, "update", teacher)["updated"] == 2


def test_csv_missing_columns(admin_scope):
    result = acs.validate_academic_csv(pd.DataFrame({"student_id": ["STU0001"], "subject": ["X"]}), admin_scope)
    assert not result.structure_ok and "attendance" in result.missing_columns


def test_template_rows_are_valid(admin_scope):
    import io
    raw = pd.read_csv(io.BytesIO(acs.template_csv(admin_scope)), dtype=str)
    result = acs.validate_academic_csv(raw, admin_scope)
    assert result.error_rows == 0
