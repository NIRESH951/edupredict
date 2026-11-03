"""
Subject catalogue management.
"""
from __future__ import annotations

import re
import sqlite3

import pandas as pd

import config
from core import database as db


class SubjectError(ValueError):
    """User-facing subject-management error."""


def list_subjects(department: str | None = None, semester: int | None = None, search: str = "") -> pd.DataFrame:
    sql = """SELECT s.id, s.subject_code, s.subject_name, s.department, s.semester, s.credits,
                    (SELECT COUNT(*) FROM academic_records r
                      JOIN students st ON st.student_id = r.student_id
                     WHERE r.subject = s.subject_name AND r.semester = s.semester
                       AND st.department = s.department) AS records
             FROM subjects s WHERE 1=1"""
    params: list = []
    if department and department != "All":
        sql += " AND s.department = ?"
        params.append(department)
    if semester and semester != "All":
        sql += " AND s.semester = ?"
        params.append(int(semester))
    if search:
        sql += " AND (s.subject_code LIKE ? OR s.subject_name LIKE ?)"
        params += [f"%{search}%", f"%{search}%"]
    sql += " ORDER BY s.department, s.semester, s.subject_code"
    return db.query_df(sql, params)


def subject_names(department: str | None = None, semester: int | None = None) -> list[str]:
    df = list_subjects(department, semester)
    return sorted(df["subject_name"].unique().tolist())


def _validate(code: str, name: str, department: str, semester: int, credits: int) -> list[str]:
    errors = []
    if not re.match(r"^[A-Za-z0-9-]{3,12}$", code or ""):
        errors.append("Subject code must be 3-12 letters/digits (e.g. AI501).")
    if not name or len(name.strip()) < 3:
        errors.append("Subject name must have at least 3 characters.")
    if department not in config.DEPARTMENTS:
        errors.append("Invalid department.")
    if not 1 <= int(semester) <= 8:
        errors.append("Semester must be between 1 and 8.")
    if not 1 <= int(credits) <= 6:
        errors.append("Credits must be between 1 and 6.")
    return errors


def add_subject(code: str, name: str, department: str, semester: int, credits: int = 4,
                acting_user: dict | None = None) -> int:
    errors = _validate(code, name, department, semester, credits)
    if errors:
        raise SubjectError("\n".join(errors))
    try:
        sid = db.execute(
            "INSERT INTO subjects (subject_code, subject_name, department, semester, credits) VALUES (?, ?, ?, ?, ?)",
            [code.strip().upper(), " ".join(name.split()), department, int(semester), int(credits)],
        )
    except sqlite3.IntegrityError as exc:
        raise SubjectError("A subject with this code (or this name in the same department/semester) already exists.") from exc
    if acting_user:
        db.log_activity(acting_user.get("id"), acting_user.get("username"), "SUBJECT_ADDED", code)
    return sid


def update_subject(subject_id: int, code: str, name: str, department: str, semester: int, credits: int,
                   acting_user: dict | None = None) -> None:
    errors = _validate(code, name, department, semester, credits)
    if errors:
        raise SubjectError("\n".join(errors))
    try:
        db.execute(
            """UPDATE subjects SET subject_code = ?, subject_name = ?, department = ?, semester = ?, credits = ?
               WHERE id = ?""",
            [code.strip().upper(), " ".join(name.split()), department, int(semester), int(credits), subject_id],
        )
    except sqlite3.IntegrityError as exc:
        raise SubjectError("Another subject already uses this code or name.") from exc
    if acting_user:
        db.log_activity(acting_user.get("id"), acting_user.get("username"), "SUBJECT_UPDATED", code)


def delete_subject(subject_id: int, acting_user: dict | None = None) -> None:
    db.execute("DELETE FROM subjects WHERE id = ?", [subject_id])
    if acting_user:
        db.log_activity(acting_user.get("id"), acting_user.get("username"), "SUBJECT_DELETED", str(subject_id))
