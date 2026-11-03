"""
Teacher management: teacher profiles and class assignments.
"""
from __future__ import annotations

import re
import sqlite3

import pandas as pd

import config
from core import database as db
from core.validation import is_blank, valid_email


class TeacherError(ValueError):
    """User-facing teacher-management error."""


def next_teacher_id() -> str:
    last = db.query_scalar(
        "SELECT teacher_id FROM teachers WHERE teacher_id LIKE 'TCH%' "
        "ORDER BY CAST(SUBSTR(teacher_id, 4) AS INTEGER) DESC LIMIT 1"
    )
    number = int(last[3:]) + 1 if last and last[3:].isdigit() else 1
    return f"TCH{number:03d}"


def _validate(data: dict) -> list[str]:
    errors = []
    name = str(data.get("full_name", "")).strip()
    if len(name) < 3:
        errors.append("Full name must have at least 3 characters.")
    if not is_blank(data.get("email")) and not valid_email(str(data["email"])):
        errors.append("Email address is not valid.")
    if data.get("department") not in config.DEPARTMENTS:
        errors.append("Please choose a valid department.")
    phone = str(data.get("phone") or "").strip()
    if phone and not re.match(r"^[+0-9 -]{7,15}$", phone):
        errors.append("Phone number may contain only digits, spaces, '+' and '-' (7-15 characters).")
    return errors


def add_teacher(data: dict, acting_user: dict | None = None) -> str:
    errors = _validate(data)
    if errors:
        raise TeacherError("\n".join(errors))
    teacher_id = next_teacher_id()
    try:
        db.execute(
            """INSERT INTO teachers (teacher_id, full_name, email, department, designation, phone)
               VALUES (?, ?, ?, ?, ?, ?)""",
            [teacher_id, " ".join(data["full_name"].split()), (data.get("email") or "").strip().lower() or None,
             data["department"], data.get("designation"), (data.get("phone") or "").strip() or None],
        )
    except sqlite3.IntegrityError as exc:
        raise TeacherError("A teacher with this email address already exists.") from exc
    if acting_user:
        db.log_activity(acting_user.get("id"), acting_user.get("username"), "TEACHER_ADDED", teacher_id)
    return teacher_id


def update_teacher(teacher_id: str, data: dict, acting_user: dict | None = None) -> None:
    errors = _validate(data)
    if errors:
        raise TeacherError("\n".join(errors))
    try:
        db.execute(
            """UPDATE teachers SET full_name = ?, email = ?, department = ?, designation = ?, phone = ?
               WHERE teacher_id = ?""",
            [" ".join(data["full_name"].split()), (data.get("email") or "").strip().lower() or None,
             data["department"], data.get("designation"), (data.get("phone") or "").strip() or None, teacher_id],
        )
    except sqlite3.IntegrityError as exc:
        raise TeacherError("A teacher with this email address already exists.") from exc
    db.execute("UPDATE users SET full_name = ? WHERE teacher_id = ?", [" ".join(data["full_name"].split()), teacher_id])
    if acting_user:
        db.log_activity(acting_user.get("id"), acting_user.get("username"), "TEACHER_UPDATED", teacher_id)


def delete_teacher(teacher_id: str, acting_user: dict | None = None) -> None:
    db.execute("DELETE FROM teachers WHERE teacher_id = ?", [teacher_id])
    if acting_user:
        db.log_activity(acting_user.get("id"), acting_user.get("username"), "TEACHER_DELETED", teacher_id)


def get_teacher(teacher_id: str) -> dict | None:
    return db.query_one("SELECT * FROM teachers WHERE teacher_id = ?", [teacher_id])


def list_teachers(search: str = "", department: str | None = None) -> pd.DataFrame:
    sql = """SELECT t.teacher_id, t.full_name, t.email, t.department, t.designation, t.phone,
                    u.username AS login_username,
                    (SELECT COUNT(*) FROM teacher_assignments ta WHERE ta.teacher_id = t.teacher_id) AS classes,
                    (SELECT COUNT(*) FROM students s JOIN teacher_assignments ta
                        ON s.department = ta.department AND s.semester = ta.semester AND s.section = ta.section
                     WHERE ta.teacher_id = t.teacher_id) AS students
             FROM teachers t LEFT JOIN users u ON u.teacher_id = t.teacher_id WHERE 1=1"""
    params: list = []
    if search:
        sql += " AND (t.teacher_id LIKE ? OR t.full_name LIKE ? OR t.email LIKE ?)"
        like = f"%{search.strip()}%"
        params += [like, like, like]
    if department and department != "All":
        sql += " AND t.department = ?"
        params.append(department)
    sql += " ORDER BY t.teacher_id"
    return db.query_df(sql, params)


def list_assignments(teacher_id: str | None = None) -> pd.DataFrame:
    sql = """SELECT ta.id, ta.teacher_id, t.full_name AS teacher, ta.department, ta.semester, ta.section,
                    (SELECT COUNT(*) FROM students s WHERE s.department = ta.department
                        AND s.semester = ta.semester AND s.section = ta.section) AS students
             FROM teacher_assignments ta JOIN teachers t ON t.teacher_id = ta.teacher_id"""
    params: list = []
    if teacher_id:
        sql += " WHERE ta.teacher_id = ?"
        params.append(teacher_id)
    sql += " ORDER BY ta.teacher_id, ta.department, ta.semester, ta.section"
    return db.query_df(sql, params)


def add_assignment(teacher_id: str, department: str, semester: int, section: str,
                   acting_user: dict | None = None) -> None:
    if department not in config.DEPARTMENTS:
        raise TeacherError("Invalid department.")
    try:
        db.execute(
            "INSERT INTO teacher_assignments (teacher_id, department, semester, section) VALUES (?, ?, ?, ?)",
            [teacher_id, department, int(semester), section.upper()],
        )
    except sqlite3.IntegrityError as exc:
        raise TeacherError("This class is already assigned to the teacher.") from exc
    if acting_user:
        db.log_activity(acting_user.get("id"), acting_user.get("username"), "CLASS_ASSIGNED",
                        f"{teacher_id} -> {department} S{semester}{section}")


def remove_assignment(assignment_id: int, acting_user: dict | None = None) -> None:
    db.execute("DELETE FROM teacher_assignments WHERE id = ?", [assignment_id])
    if acting_user:
        db.log_activity(acting_user.get("id"), acting_user.get("username"), "CLASS_UNASSIGNED", str(assignment_id))


def teachers_without_login() -> pd.DataFrame:
    return db.query_df(
        """SELECT t.teacher_id, t.full_name, t.email, t.department FROM teachers t
           LEFT JOIN users u ON u.teacher_id = t.teacher_id WHERE u.id IS NULL ORDER BY t.teacher_id"""
    )
