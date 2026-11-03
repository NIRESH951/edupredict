"""
Input validation shared by the forms, the CSV importer and the services.
"""
from __future__ import annotations

import math
import re
from datetime import date, datetime
from typing import Any

import pandas as pd

import config

EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")
REG_NO_RE = re.compile(r"^[A-Za-z0-9\-/]{4,20}$")
ACADEMIC_YEAR_RE = re.compile(r"^(\d{4})-(\d{2})$")
USERNAME_RE = re.compile(r"^[A-Za-z0-9._-]{3,30}$")


def is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return isinstance(value, str) and value.strip() == ""


def to_float(value: Any) -> float | None:
    if is_blank(value):
        return None
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def to_int(value: Any) -> int | None:
    f = to_float(value)
    if f is None or not float(f).is_integer():
        return None
    return int(f)


def valid_email(email: str | None) -> bool:
    return bool(email) and bool(EMAIL_RE.match(email.strip()))


def valid_username(username: str) -> bool:
    return bool(USERNAME_RE.match(username or ""))


def valid_academic_year(value: str) -> bool:
    """Accepts 'YYYY-YY' where YY is the following year (e.g. 2026-27)."""
    m = ACADEMIC_YEAR_RE.match((value or "").strip())
    if not m:
        return False
    start, end = int(m.group(1)), int(m.group(2))
    return (start + 1) % 100 == end


def validate_numeric_field(field: str, value: Any, *, required: bool = True) -> tuple[float | None, str | None]:
    """Return (parsed_value, error_message)."""
    label = config.FEATURE_LABELS.get(field, field)
    if is_blank(value):
        return (None, f"{label} is required") if required else (None, None)
    number = to_float(value)
    if number is None:
        return None, f"{label} must be a number (got '{value}')"
    lo, hi = config.FIELD_LIMITS[field]
    if not lo <= number <= hi:
        return None, f"{label} must be between {lo:g} and {hi:g} (got {number:g})"
    return round(number, 2), None


def validate_academic_record(data: dict) -> tuple[dict, list[str]]:
    """
    Validate one academic record. Returns (clean_record, errors).
    `final_marks` is optional because records are created before the exam.
    """
    errors: list[str] = []
    clean: dict[str, Any] = {}

    student_id = "" if is_blank(data.get("student_id")) else str(data.get("student_id")).strip().upper()
    if not student_id:
        errors.append("Student ID is required")
    clean["student_id"] = student_id

    subject = "" if is_blank(data.get("subject")) else re.sub(r"\s+", " ", str(data.get("subject")).strip())
    if not subject:
        errors.append("Subject is required")
    elif len(subject) > 100:
        errors.append("Subject name is too long (max 100 characters)")
    clean["subject"] = subject

    semester = to_int(data.get("semester"))
    if semester is None or not 1 <= semester <= 8:
        errors.append("Semester must be a whole number between 1 and 8")
    clean["semester"] = semester

    for field in config.FEATURE_COLUMNS:
        value, err = validate_numeric_field(field, data.get(field), required=True)
        clean[field] = value
        if err:
            errors.append(err)

    final, err = validate_numeric_field("final_marks", data.get("final_marks"), required=False)
    clean["final_marks"] = final
    if err:
        errors.append(err)

    return clean, errors


def validate_student(data: dict) -> list[str]:
    errors = []
    if is_blank(data.get("reg_no")) or not REG_NO_RE.match(str(data["reg_no"]).strip()):
        errors.append("Registration number must be 4-20 letters/digits (e.g. 1EP23AI001)")
    if is_blank(data.get("full_name")) or len(str(data["full_name"]).strip()) < 3:
        errors.append("Full name must have at least 3 characters")
    elif not re.match(r"^[A-Za-z .'-]+$", str(data["full_name"]).strip()):
        errors.append("Full name may contain only letters, spaces, dots, apostrophes and hyphens")
    if not is_blank(data.get("email")) and not valid_email(str(data["email"])):
        errors.append("Email address is not valid")
    if data.get("department") not in config.DEPARTMENTS:
        errors.append("Please choose a valid department")
    sem = to_int(data.get("semester"))
    if sem is None or not 1 <= sem <= 8:
        errors.append("Semester must be between 1 and 8")
    if not valid_academic_year(str(data.get("academic_year", ""))):
        errors.append("Academic year must look like 2026-27")
    if is_blank(data.get("section")) or not re.match(r"^[A-Za-z]$", str(data["section"]).strip()):
        errors.append("Section must be a single letter")
    enrol = data.get("enrollment_date")
    if isinstance(enrol, str):
        try:
            enrol = datetime.strptime(enrol, "%Y-%m-%d").date()
        except ValueError:
            errors.append("Enrollment date must be in YYYY-MM-DD format")
            enrol = None
    if isinstance(enrol, date) and enrol > date.today():
        errors.append("Enrollment date cannot be in the future")
    elif enrol is None and "Enrollment date must be in YYYY-MM-DD format" not in errors:
        errors.append("Enrollment date is required")
    return errors


def normalise_column_name(name: str) -> str:
    """'Internal Test 1 ' -> 'internal_test_1'"""
    name = str(name).strip().lower()
    name = re.sub(r"[^a-z0-9%]+", "_", name).strip("_")
    return name


def map_columns(columns: list[str]) -> dict[str, str]:
    """Map raw CSV headers to canonical column names using config.COLUMN_ALIASES."""
    mapping: dict[str, str] = {}
    lookup: dict[str, str] = {}
    for canonical, aliases in config.COLUMN_ALIASES.items():
        lookup[canonical] = canonical
        for alias in aliases:
            lookup[normalise_column_name(alias)] = canonical
    used = set()
    for col in columns:
        key = normalise_column_name(col)
        canonical = lookup.get(key)
        if canonical and canonical not in used:
            mapping[col] = canonical
            used.add(canonical)
    return mapping
