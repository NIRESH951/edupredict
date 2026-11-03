"""
Student management (Module 3): CRUD, search/filter and login assignment.
"""
from __future__ import annotations

import sqlite3
from datetime import date, datetime

import pandas as pd

from core import database as db
from core.scope import AccessDenied, Scope
from core.validation import validate_student


class StudentError(ValueError):
    """User-facing student-management error."""


def next_student_id() -> str:
    last = db.query_scalar(
        "SELECT student_id FROM students WHERE student_id LIKE 'STU%' "
        "ORDER BY CAST(SUBSTR(student_id, 4) AS INTEGER) DESC LIMIT 1"
    )
    number = int(last[3:]) + 1 if last and last[3:].isdigit() else 1
    return f"STU{number:04d}"


def _clean(data: dict) -> dict:
    enrol = data.get("enrollment_date")
    if isinstance(enrol, (date, datetime)):
        enrol = enrol.strftime("%Y-%m-%d")
    email = (data.get("email") or "").strip().lower() or None
    return {
        "reg_no": str(data.get("reg_no", "")).strip().upper(),
        "full_name": " ".join(str(data.get("full_name", "")).split()).title(),
        "email": email,
        "department": data.get("department"),
        "semester": int(data.get("semester")),
        "academic_year": str(data.get("academic_year", "")).strip(),
        "section": str(data.get("section", "")).strip().upper(),
        "enrollment_date": enrol,
    }


def _integrity_message(exc: sqlite3.IntegrityError) -> str:
    msg = str(exc).lower()
    if "reg_no" in msg:
        return "A student with this registration number already exists."
    if "email" in msg:
        return "A student with this email address already exists."
    return f"Could not save student: {exc}"


def add_student(data: dict, acting_user: dict | None = None) -> str:
    errors = validate_student(data)
    if errors:
        raise StudentError("\n".join(errors))
    clean = _clean(data)
    student_id = (data.get("student_id") or "").strip().upper() or next_student_id()
    try:
        db.execute(
            """INSERT INTO students (student_id, reg_no, full_name, email, department, semester,
                                     academic_year, section, enrollment_date)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [student_id, *clean.values()],
        )
    except sqlite3.IntegrityError as exc:
        if "student_id" in str(exc).lower():
            raise StudentError(f"Student ID {student_id} already exists.") from exc
        raise StudentError(_integrity_message(exc)) from exc
    if acting_user:
        db.log_activity(acting_user.get("id"), acting_user.get("username"), "STUDENT_ADDED",
                        f"{student_id} {clean['reg_no']}")
    return student_id


def update_student(student_id: str, data: dict, scope: Scope, acting_user: dict | None = None) -> None:
    if not scope.is_admin:
        raise AccessDenied("Only administrators can edit student profiles.")
    errors = validate_student(data)
    if errors:
        raise StudentError("\n".join(errors))
    clean = _clean(data)
    try:
        db.execute(
            """UPDATE students SET reg_no = ?, full_name = ?, email = ?, department = ?, semester = ?,
                      academic_year = ?, section = ?, enrollment_date = ?,
                      updated_at = datetime('now', 'localtime')
               WHERE student_id = ?""",
            [*clean.values(), student_id],
        )
    except sqlite3.IntegrityError as exc:
        raise StudentError(_integrity_message(exc)) from exc
    # Keep the linked login account's display name in sync.
    db.execute("UPDATE users SET full_name = ?, email = COALESCE(?, email) WHERE student_id = ?",
               [clean["full_name"], clean["email"], student_id])
    if acting_user:
        db.log_activity(acting_user.get("id"), acting_user.get("username"), "STUDENT_UPDATED", student_id)


def deletion_impact(student_id: str) -> dict:
    return {
        "records": int(db.query_scalar("SELECT COUNT(*) FROM academic_records WHERE student_id = ?", [student_id], 0)),
        "predictions": int(db.query_scalar("SELECT COUNT(*) FROM predictions WHERE student_id = ?", [student_id], 0)),
        "accounts": db.query_all("SELECT username FROM users WHERE student_id = ?", [student_id]),
    }


def delete_student(student_id: str, confirm_reg_no: str, scope: Scope, acting_user: dict | None = None) -> dict:
    """Delete a student and (via ON DELETE CASCADE) their records, predictions and login."""
    if not scope.is_admin:
        raise AccessDenied("Only administrators can delete students.")
    student = get_student(student_id, scope)
    if student is None:
        raise StudentError("Student not found.")
    if (confirm_reg_no or "").strip().upper() != student["reg_no"].upper():
        raise StudentError("Confirmation failed: the registration number you typed does not match.")
    impact = deletion_impact(student_id)
    db.execute("DELETE FROM students WHERE student_id = ?", [student_id])
    if acting_user:
        db.log_activity(acting_user.get("id"), acting_user.get("username"), "STUDENT_DELETED",
                        f"{student_id} {student['reg_no']} ({impact['records']} records)")
    return impact


def get_student(student_id: str, scope: Scope) -> dict | None:
    clause, params = scope.student_filter("s.student_id")
    return db.query_one(
        f"""SELECT s.*, u.username AS login_username, u.is_active AS login_active
            FROM students s LEFT JOIN users u ON u.student_id = s.student_id
            WHERE s.student_id = ? AND {clause}""",
        [student_id, *params],
    )


def list_students(
    scope: Scope,
    search: str = "",
    department: str | None = None,
    semester: int | None = None,
    section: str | None = None,
) -> pd.DataFrame:
    clause, params = scope.student_filter("s.student_id")
    sql = f"""SELECT s.student_id, s.reg_no, s.full_name, s.email, s.department, s.semester,
                     s.academic_year, s.section, s.enrollment_date,
                     u.username AS login_username,
                     (SELECT COUNT(*) FROM academic_records r WHERE r.student_id = s.student_id) AS records
              FROM students s LEFT JOIN users u ON u.student_id = s.student_id
              WHERE {clause}"""
    if search:
        sql += " AND (s.student_id LIKE ? OR s.reg_no LIKE ? OR s.full_name LIKE ? OR s.email LIKE ?)"
        like = f"%{search.strip()}%"
        params += [like, like, like, like]
    if department and department != "All":
        sql += " AND s.department = ?"
        params.append(department)
    if semester and semester != "All":
        sql += " AND s.semester = ?"
        params.append(int(semester))
    if section and section != "All":
        sql += " AND s.section = ?"
        params.append(section)
    sql += " ORDER BY s.department, s.semester, s.section, s.reg_no"
    return db.query_df(sql, params)


def student_options(scope: Scope) -> dict[str, str]:
    """{student_id: 'STU0001 · 1EP23AI001 · Name'} for select boxes."""
    df = list_students(scope)
    return {
        r.student_id: f"{r.student_id} · {r.reg_no} · {r.full_name}"
        for r in df.itertuples()
    }


def students_without_login() -> pd.DataFrame:
    return db.query_df(
        """SELECT s.student_id, s.reg_no, s.full_name, s.email, s.department, s.semester, s.section
           FROM students s LEFT JOIN users u ON u.student_id = s.student_id
           WHERE u.id IS NULL ORDER BY s.reg_no"""
    )


def suggested_username(student: dict) -> str:
    return student["reg_no"].lower()
