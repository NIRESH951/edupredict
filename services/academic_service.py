"""
Academic data management (Module 4): records CRUD, search and CSV import.
"""
from __future__ import annotations

import io
import sqlite3
from dataclasses import dataclass, field

import pandas as pd

import config
from core import database as db
from core.scope import AccessDenied, Scope
from core.validation import map_columns, validate_academic_record

RECORD_COLUMNS = ["student_id", "subject", "semester", *config.FEATURE_COLUMNS, "final_marks"]


class RecordError(ValueError):
    """User-facing academic-record error."""


# --------------------------------------------------------------------------- #
# Queries
# --------------------------------------------------------------------------- #
def list_records(
    scope: Scope,
    search: str = "",
    department: str | None = None,
    semester: int | None = None,
    section: str | None = None,
    subject: str | None = None,
    student_id: str | None = None,
    status: str = "All",
) -> pd.DataFrame:
    """Academic records joined with student details and the latest prediction."""
    clause, params = scope.student_filter("r.student_id")
    sql = f"""
        SELECT r.id, r.student_id, s.reg_no, s.full_name, s.department, s.section,
               s.semester AS current_semester, r.subject, r.semester,
               r.attendance, r.internal1, r.internal2, r.assignment, r.study_hours, r.previous_gpa,
               r.final_marks, lp.predicted_marks, lp.baseline_marks, lp.performance_category,
               lp.risk_level, lp.risk_confidence, lp.prob_low, lp.prob_moderate, lp.prob_high,
               lp.model_version, lp.created_at AS predicted_at, r.updated_at, r.created_at
        FROM academic_records r
        JOIN students s ON s.student_id = r.student_id
        LEFT JOIN latest_predictions lp ON lp.record_id = r.id
        WHERE {clause}"""
    if search:
        sql += " AND (r.student_id LIKE ? OR s.reg_no LIKE ? OR s.full_name LIKE ? OR r.subject LIKE ?)"
        like = f"%{search.strip()}%"
        params += [like, like, like, like]
    if department and department != "All":
        sql += " AND s.department = ?"
        params.append(department)
    if semester and semester != "All":
        sql += " AND r.semester = ?"
        params.append(int(semester))
    if section and section != "All":
        sql += " AND s.section = ?"
        params.append(section)
    if subject and subject != "All":
        sql += " AND r.subject = ?"
        params.append(subject)
    if student_id:
        sql += " AND r.student_id = ?"
        params.append(student_id)
    if status == "Awaiting final exam":
        sql += " AND r.final_marks IS NULL"
    elif status == "Final marks available":
        sql += " AND r.final_marks IS NOT NULL"
    sql += " ORDER BY s.reg_no, r.semester, r.subject"
    df = db.query_df(sql, params)
    if not df.empty:
        df["internal_avg_pct"] = ((df["internal1"] + df["internal2"]) / 100 * 100).round(1)
    return df


def get_record(record_id: int, scope: Scope) -> dict | None:
    clause, params = scope.student_filter("student_id")
    return db.query_one(f"SELECT * FROM academic_records WHERE id = ? AND {clause}", [record_id, *params])


def subjects_in_records(scope: Scope) -> list[str]:
    clause, params = scope.student_filter("student_id")
    df = db.query_df(f"SELECT DISTINCT subject FROM academic_records WHERE {clause} ORDER BY subject", params)
    return df["subject"].tolist()


# --------------------------------------------------------------------------- #
# Single-record CRUD
# --------------------------------------------------------------------------- #
def _check_semester(student_id: str, semester: int) -> str | None:
    current = db.query_scalar("SELECT semester FROM students WHERE student_id = ?", [student_id])
    if current is None:
        return f"Student ID {student_id} does not exist."
    if semester > int(current):
        return f"Semester {semester} is after the student's current semester ({current})."
    return None


def add_record(data: dict, scope: Scope, acting_user: dict | None = None) -> int:
    if not scope.can_edit_records:
        raise AccessDenied("You are not allowed to add academic records.")
    clean, errors = validate_academic_record(data)
    if errors:
        raise RecordError("\n".join(errors))
    scope.ensure_student_access(clean["student_id"])
    sem_error = _check_semester(clean["student_id"], clean["semester"])
    if sem_error:
        raise RecordError(sem_error)
    try:
        record_id = db.execute(
            f"""INSERT INTO academic_records ({", ".join(RECORD_COLUMNS)}, created_by)
                VALUES ({", ".join("?" * len(RECORD_COLUMNS))}, ?)""",
            [clean[c] for c in RECORD_COLUMNS] + [acting_user.get("id") if acting_user else None],
        )
    except sqlite3.IntegrityError as exc:
        if "unique" in str(exc).lower():
            raise RecordError(
                f"A record for {clean['student_id']} / {clean['subject']} / semester {clean['semester']} already exists."
            ) from exc
        raise RecordError(f"Could not save record: {exc}") from exc
    if acting_user:
        db.log_activity(acting_user.get("id"), acting_user.get("username"), "RECORD_ADDED",
                        f"{clean['student_id']} {clean['subject']} S{clean['semester']}")
    return record_id


def update_record(record_id: int, data: dict, scope: Scope, acting_user: dict | None = None) -> bool:
    """Update a record. Returns True if model input features changed (prediction now stale)."""
    if not scope.can_edit_records:
        raise AccessDenied("You are not allowed to edit academic records.")
    existing = get_record(record_id, scope)
    if existing is None:
        raise AccessDenied("Record not found or not accessible.")
    clean, errors = validate_academic_record({**existing, **data})
    if errors:
        raise RecordError("\n".join(errors))
    scope.ensure_student_access(clean["student_id"])
    sem_error = _check_semester(clean["student_id"], clean["semester"])
    if sem_error:
        raise RecordError(sem_error)
    features_changed = any(float(existing[c]) != float(clean[c]) for c in config.FEATURE_COLUMNS)
    try:
        db.execute(
            f"""UPDATE academic_records SET {", ".join(f"{c} = ?" for c in RECORD_COLUMNS)},
                       updated_at = datetime('now', 'localtime') WHERE id = ?""",
            [clean[c] for c in RECORD_COLUMNS] + [record_id],
        )
    except sqlite3.IntegrityError as exc:
        raise RecordError("Another record already exists for this student, subject and semester.") from exc
    if features_changed:
        # Old predictions were made from different inputs and are no longer valid.
        db.execute("DELETE FROM predictions WHERE record_id = ?", [record_id])
    if acting_user:
        db.log_activity(acting_user.get("id"), acting_user.get("username"), "RECORD_UPDATED", str(record_id))
    return features_changed


def delete_record(record_id: int, scope: Scope, acting_user: dict | None = None) -> None:
    if not scope.can_edit_records:
        raise AccessDenied("You are not allowed to delete academic records.")
    if get_record(record_id, scope) is None:
        raise AccessDenied("Record not found or not accessible.")
    db.execute("DELETE FROM academic_records WHERE id = ?", [record_id])
    if acting_user:
        db.log_activity(acting_user.get("id"), acting_user.get("username"), "RECORD_DELETED", str(record_id))


# --------------------------------------------------------------------------- #
# CSV import
# --------------------------------------------------------------------------- #
@dataclass
class CsvValidationResult:
    total_rows: int = 0
    valid: pd.DataFrame = field(default_factory=pd.DataFrame)
    errors: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=["row", "student_id", "issue"]))
    warnings: list[str] = field(default_factory=list)
    missing_columns: list[str] = field(default_factory=list)
    ignored_columns: list[str] = field(default_factory=list)
    column_mapping: dict[str, str] = field(default_factory=dict)
    missing_values: pd.DataFrame = field(default_factory=pd.DataFrame)
    duplicates_in_file: int = 0
    existing_in_db: int = 0

    @property
    def structure_ok(self) -> bool:
        return not self.missing_columns

    @property
    def error_rows(self) -> int:
        return int(self.errors["row"].nunique()) if not self.errors.empty else 0


def read_csv(file) -> pd.DataFrame:
    """Read an uploaded CSV, trying UTF-8 then Latin-1. Everything is read as text first."""
    raw = file.getvalue() if hasattr(file, "getvalue") else file.read()
    if isinstance(raw, str):
        raw = raw.encode("utf-8")
    if not raw.strip():
        raise RecordError("The uploaded file is empty.")
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(raw), dtype=str, encoding=encoding, skip_blank_lines=True)
        except UnicodeDecodeError:
            continue
        except pd.errors.ParserError as exc:
            raise RecordError(f"Could not parse the CSV file: {exc}") from exc
        except pd.errors.EmptyDataError as exc:
            raise RecordError("The uploaded file has no columns.") from exc
    raise RecordError("Could not decode the file. Please save it as UTF-8 CSV.")


def validate_academic_csv(raw: pd.DataFrame, scope: Scope) -> CsvValidationResult:
    result = CsvValidationResult(total_rows=len(raw))
    mapping = map_columns(list(raw.columns))
    result.column_mapping = mapping
    result.ignored_columns = [c for c in raw.columns if c not in mapping]
    present = set(mapping.values())
    result.missing_columns = [c for c in config.ACADEMIC_CSV_REQUIRED if c not in present]
    if result.missing_columns:
        return result

    df = raw.rename(columns=mapping)[[c for c in config.ACADEMIC_CSV_COLUMNS if c in present]].copy()
    if "final_marks" not in df.columns:
        df["final_marks"] = None
    df = df.dropna(how="all")
    result.total_rows = len(df)

    # Missing-value summary (blank strings count as missing).
    blank = df.apply(lambda col: col.isna() | (col.astype(str).str.strip() == ""))
    result.missing_values = pd.DataFrame({
        "column": df.columns,
        "missing": [int(blank[c].sum()) for c in df.columns],
        "required": ["no" if c == "final_marks" else "yes" for c in df.columns],
    })

    # Students that exist / are accessible.
    all_ids = set(db.query_df("SELECT student_id FROM students")["student_id"])
    clause, params = scope.student_filter("student_id")
    students = db.query_df(f"SELECT student_id, semester FROM students WHERE {clause}", params)
    scoped_sem = dict(zip(students["student_id"], students["semester"]))
    existing = db.query_df(
        f"SELECT student_id, LOWER(subject) AS subject_key, semester FROM academic_records WHERE {clause}", params
    )
    existing_keys = {(r.student_id, r.subject_key, int(r.semester)) for r in existing.itertuples()}
    catalogue = {s.lower() for s in db.query_df("SELECT subject_name FROM subjects")["subject_name"]}

    errors: list[dict] = []
    clean_rows: list[dict] = []
    seen: dict[tuple, int] = {}
    unknown_subjects: set[str] = set()

    for idx, row in enumerate(df.to_dict("records")):
        row_no = idx + 2  # +1 for header, +1 for 1-based numbering
        clean, row_errors = validate_academic_record(row)
        sid = clean.get("student_id") or ""
        if sid and not row_errors:
            if sid not in all_ids:
                row_errors.append(f"Student ID {sid} does not exist")
            elif sid not in scoped_sem:
                row_errors.append(f"Student {sid} is not assigned to you")
            elif clean["semester"] > int(scoped_sem[sid]):
                row_errors.append(f"Semester {clean['semester']} is after the student's current semester ({scoped_sem[sid]})")
        if not row_errors:
            key = (sid, clean["subject"].lower(), clean["semester"])
            if key in seen:
                row_errors.append(f"Duplicate of row {seen[key]} in this file")
                result.duplicates_in_file += 1
            else:
                seen[key] = row_no
        for msg in row_errors:
            errors.append({"row": row_no, "student_id": sid, "issue": msg})
        if row_errors:
            continue
        clean["row"] = row_no
        clean["exists"] = (sid, clean["subject"].lower(), clean["semester"]) in existing_keys
        if clean["subject"].lower() not in catalogue:
            unknown_subjects.add(clean["subject"])
        clean_rows.append(clean)

    result.errors = pd.DataFrame(errors, columns=["row", "student_id", "issue"])
    result.valid = pd.DataFrame(clean_rows)
    result.existing_in_db = int(result.valid["exists"].sum()) if not result.valid.empty else 0
    if unknown_subjects:
        names = ", ".join(sorted(unknown_subjects)[:8])
        more = "" if len(unknown_subjects) <= 8 else f" and {len(unknown_subjects) - 8} more"
        result.warnings.append(f"Subjects not in the subject catalogue (will still be imported): {names}{more}.")
    if result.ignored_columns:
        result.warnings.append("Ignored unrecognised columns: " + ", ".join(map(str, result.ignored_columns)) + ".")
    return result


def import_academic_records(valid: pd.DataFrame, scope: Scope, mode: str = "skip",
                            acting_user: dict | None = None) -> dict:
    """
    Import validated rows. mode='skip' leaves existing records untouched,
    mode='update' overwrites them. Returns counts and the affected record IDs.
    """
    if not scope.can_edit_records:
        raise AccessDenied("You are not allowed to import academic records.")
    counts = {"inserted": 0, "updated": 0, "skipped": 0, "record_ids": []}
    if valid.empty:
        return counts
    user_id = acting_user.get("id") if acting_user else None
    with db.get_db() as conn:
        for row in valid.to_dict("records"):
            if not scope.can_access_student(row["student_id"]):
                counts["skipped"] += 1
                continue
            final = None if pd.isna(row.get("final_marks")) else row.get("final_marks")
            values = [row[c] for c in RECORD_COLUMNS[:-1]] + [final]
            found = conn.execute(
                "SELECT id, " + ", ".join(config.FEATURE_COLUMNS) +
                " FROM academic_records WHERE student_id = ? AND LOWER(subject) = LOWER(?) AND semester = ?",
                [row["student_id"], row["subject"], row["semester"]],
            ).fetchone()
            if found:
                if mode != "update":
                    counts["skipped"] += 1
                    continue
                conn.execute(
                    f"""UPDATE academic_records SET {", ".join(f"{c} = ?" for c in RECORD_COLUMNS)},
                               updated_at = datetime('now', 'localtime') WHERE id = ?""",
                    values + [found["id"]],
                )
                if any(float(found[c]) != float(row[c]) for c in config.FEATURE_COLUMNS):
                    conn.execute("DELETE FROM predictions WHERE record_id = ?", [found["id"]])
                counts["updated"] += 1
                counts["record_ids"].append(found["id"])
            else:
                cur = conn.execute(
                    f"""INSERT INTO academic_records ({", ".join(RECORD_COLUMNS)}, created_by)
                        VALUES ({", ".join("?" * len(RECORD_COLUMNS))}, ?)""",
                    values + [user_id],
                )
                counts["inserted"] += 1
                counts["record_ids"].append(cur.lastrowid)
    if acting_user:
        db.log_activity(user_id, acting_user.get("username"), "CSV_IMPORT",
                        f"inserted={counts['inserted']} updated={counts['updated']} skipped={counts['skipped']}")
    return counts


def template_csv(scope: Scope | None = None) -> bytes:
    """CSV template with headers and two example rows that are valid for students in the user's scope."""
    scope = scope or Scope.admin()
    clause, params = scope.student_filter("s.student_id")
    picks = db.query_df(
        f"""SELECT s.student_id, s.semester,
                   (SELECT subject_name FROM subjects sub WHERE sub.department = s.department
                     AND sub.semester = s.semester ORDER BY sub.subject_code LIMIT 1) AS subject
            FROM students s WHERE {clause} ORDER BY s.student_id LIMIT 2""",
        params,
    )
    examples = [(r.student_id, r.subject or "Machine Learning", int(r.semester)) for r in picks.itertuples()]
    while len(examples) < 2:
        examples.append(("STU0001", "Machine Learning", 5))
    rows = [
        [examples[0][0], examples[0][1], examples[0][2], 86.5, 38, 41, 17, 9, 8.2, ""],
        [examples[1][0], examples[1][1], examples[1][2], 71.0, 27, 30.5, 14, 5, 6.9, 64],
    ]
    df = pd.DataFrame(rows, columns=config.ACADEMIC_CSV_COLUMNS)
    return df.to_csv(index=False).encode("utf-8")


def export_records_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")
