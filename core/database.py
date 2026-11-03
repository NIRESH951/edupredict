"""
SQLite data-access layer.

All SQL goes through the helpers in this module, which always use
parameterised queries (no string-formatted user input) and enable
foreign-key enforcement on every connection.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from typing import Any, Iterable, Iterator

import pandas as pd

import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS students (
    student_id       TEXT PRIMARY KEY,
    reg_no           TEXT NOT NULL UNIQUE COLLATE NOCASE,
    full_name        TEXT NOT NULL,
    email            TEXT UNIQUE COLLATE NOCASE,
    department       TEXT NOT NULL,
    semester         INTEGER NOT NULL CHECK (semester BETWEEN 1 AND 8),
    academic_year    TEXT NOT NULL,
    section          TEXT NOT NULL,
    enrollment_date  TEXT NOT NULL,
    created_at       TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
    updated_at       TEXT
);

CREATE TABLE IF NOT EXISTS teachers (
    teacher_id   TEXT PRIMARY KEY,
    full_name    TEXT NOT NULL,
    email        TEXT UNIQUE COLLATE NOCASE,
    department   TEXT NOT NULL,
    designation  TEXT,
    phone        TEXT,
    created_at   TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
);

CREATE TABLE IF NOT EXISTS teacher_assignments (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    teacher_id  TEXT NOT NULL REFERENCES teachers(teacher_id) ON DELETE CASCADE,
    department  TEXT NOT NULL,
    semester    INTEGER NOT NULL CHECK (semester BETWEEN 1 AND 8),
    section     TEXT NOT NULL,
    UNIQUE (teacher_id, department, semester, section)
);

CREATE TABLE IF NOT EXISTS users (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    username              TEXT NOT NULL UNIQUE COLLATE NOCASE,
    password_hash         TEXT NOT NULL,
    role                  TEXT NOT NULL CHECK (role IN ('admin', 'teacher', 'student')),
    full_name             TEXT NOT NULL,
    email                 TEXT,
    is_active             INTEGER NOT NULL DEFAULT 1,
    student_id            TEXT REFERENCES students(student_id) ON DELETE CASCADE,
    teacher_id            TEXT REFERENCES teachers(teacher_id) ON DELETE CASCADE,
    must_change_password  INTEGER NOT NULL DEFAULT 0,
    failed_attempts       INTEGER NOT NULL DEFAULT 0,
    locked_until          TEXT,
    created_at            TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
    last_login            TEXT
);

CREATE TABLE IF NOT EXISTS subjects (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    subject_code  TEXT NOT NULL UNIQUE COLLATE NOCASE,
    subject_name  TEXT NOT NULL,
    department    TEXT NOT NULL,
    semester      INTEGER NOT NULL CHECK (semester BETWEEN 1 AND 8),
    credits       INTEGER NOT NULL DEFAULT 4,
    UNIQUE (subject_name, department, semester)
);

CREATE TABLE IF NOT EXISTS academic_records (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id    TEXT NOT NULL REFERENCES students(student_id) ON DELETE CASCADE,
    subject       TEXT NOT NULL COLLATE NOCASE,
    semester      INTEGER NOT NULL CHECK (semester BETWEEN 1 AND 8),
    attendance    REAL NOT NULL CHECK (attendance BETWEEN 0 AND 100),
    internal1     REAL NOT NULL CHECK (internal1 BETWEEN 0 AND 50),
    internal2     REAL NOT NULL CHECK (internal2 BETWEEN 0 AND 50),
    assignment    REAL NOT NULL CHECK (assignment BETWEEN 0 AND 20),
    study_hours   REAL NOT NULL CHECK (study_hours BETWEEN 0 AND 60),
    previous_gpa  REAL NOT NULL CHECK (previous_gpa BETWEEN 0 AND 10),
    final_marks   REAL CHECK (final_marks IS NULL OR final_marks BETWEEN 0 AND 100),
    created_by    INTEGER REFERENCES users(id) ON DELETE SET NULL,
    created_at    TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
    updated_at    TEXT,
    UNIQUE (student_id, subject, semester)
);

-- Uploaded historical datasets used only to train models (no student link required).
CREATE TABLE IF NOT EXISTS datasets (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT NOT NULL,
    filename      TEXT,
    description   TEXT,
    row_count     INTEGER NOT NULL DEFAULT 0,
    rejected_rows INTEGER NOT NULL DEFAULT 0,
    include_in_training INTEGER NOT NULL DEFAULT 1,
    uploaded_by   INTEGER REFERENCES users(id) ON DELETE SET NULL,
    uploaded_at   TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
);

CREATE TABLE IF NOT EXISTS historical_records (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id    INTEGER NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    subject       TEXT,
    semester      INTEGER,
    attendance    REAL NOT NULL,
    internal1     REAL NOT NULL,
    internal2     REAL NOT NULL,
    assignment    REAL NOT NULL,
    study_hours   REAL NOT NULL,
    previous_gpa  REAL NOT NULL,
    final_marks   REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS model_registry (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    version       TEXT NOT NULL UNIQUE,
    model_dir     TEXT NOT NULL,
    n_samples     INTEGER NOT NULL,
    n_train       INTEGER NOT NULL,
    n_test        INTEGER NOT NULL,
    reg_primary_r2   REAL,
    reg_primary_mae  REAL,
    clf_primary_acc  REAL,
    clf_primary_f1   REAL,
    metrics_json  TEXT NOT NULL,
    is_active     INTEGER NOT NULL DEFAULT 0,
    trained_by    INTEGER REFERENCES users(id) ON DELETE SET NULL,
    trained_at    TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
);

CREATE TABLE IF NOT EXISTS predictions (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    record_id             INTEGER NOT NULL REFERENCES academic_records(id) ON DELETE CASCADE,
    student_id            TEXT NOT NULL REFERENCES students(student_id) ON DELETE CASCADE,
    predicted_marks       REAL NOT NULL,
    baseline_marks        REAL,
    performance_category  TEXT NOT NULL,
    risk_level            TEXT NOT NULL,
    risk_confidence       REAL,
    prob_low              REAL,
    prob_moderate         REAL,
    prob_high             REAL,
    model_version         TEXT NOT NULL,
    predicted_by          INTEGER REFERENCES users(id) ON DELETE SET NULL,
    created_at            TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
);

CREATE TABLE IF NOT EXISTS activity_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER REFERENCES users(id) ON DELETE SET NULL,
    username    TEXT,
    action      TEXT NOT NULL,
    details     TEXT,
    created_at  TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
);

CREATE INDEX IF NOT EXISTS idx_records_student ON academic_records(student_id);
CREATE INDEX IF NOT EXISTS idx_records_semester ON academic_records(semester);
CREATE INDEX IF NOT EXISTS idx_predictions_record ON predictions(record_id);
CREATE INDEX IF NOT EXISTS idx_students_class ON students(department, semester, section);
CREATE INDEX IF NOT EXISTS idx_historical_dataset ON historical_records(dataset_id);

-- Most recent prediction for every academic record.
CREATE VIEW IF NOT EXISTS latest_predictions AS
SELECT p.*
FROM predictions p
JOIN (SELECT record_id, MAX(id) AS max_id FROM predictions GROUP BY record_id) latest
  ON p.id = latest.max_id;
"""


def get_connection() -> sqlite3.Connection:
    """Open a new connection with foreign keys enabled and dict-like rows."""
    conn = sqlite3.connect(config.DB_PATH, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def get_db() -> Iterator[sqlite3.Connection]:
    """Context manager that commits on success and rolls back on error."""
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """Create all tables, indexes and views if they do not exist."""
    config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with get_db() as conn:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.executescript(SCHEMA)


def query_df(sql: str, params: Iterable[Any] = ()) -> pd.DataFrame:
    """Run a SELECT and return a pandas DataFrame."""
    with get_db() as conn:
        return pd.read_sql_query(sql, conn, params=list(params))


def query_all(sql: str, params: Iterable[Any] = ()) -> list[dict]:
    with get_db() as conn:
        return [dict(r) for r in conn.execute(sql, list(params)).fetchall()]


def query_one(sql: str, params: Iterable[Any] = ()) -> dict | None:
    with get_db() as conn:
        row = conn.execute(sql, list(params)).fetchone()
        return dict(row) if row else None


def query_scalar(sql: str, params: Iterable[Any] = (), default: Any = None) -> Any:
    with get_db() as conn:
        row = conn.execute(sql, list(params)).fetchone()
        if row is None or row[0] is None:
            return default
        return row[0]


def execute(sql: str, params: Iterable[Any] = ()) -> int:
    """Run an INSERT/UPDATE/DELETE; returns lastrowid (or rowcount for updates)."""
    with get_db() as conn:
        cur = conn.execute(sql, list(params))
        return cur.lastrowid if cur.lastrowid else cur.rowcount


def execute_many(sql: str, rows: list[Iterable[Any]]) -> int:
    with get_db() as conn:
        cur = conn.executemany(sql, [list(r) for r in rows])
        return cur.rowcount


def table_count(table: str) -> int:
    allowed = {
        "students", "teachers", "users", "subjects", "academic_records",
        "predictions", "datasets", "historical_records", "model_registry",
        "teacher_assignments", "activity_log",
    }
    if table not in allowed:
        raise ValueError(f"Unknown table: {table}")
    return int(query_scalar(f"SELECT COUNT(*) FROM {table}", default=0))


def log_activity(user_id: int | None, username: str | None, action: str, details: str = "") -> None:
    """Write an audit-trail entry. Never raises (logging must not break the app)."""
    try:
        execute(
            "INSERT INTO activity_log (user_id, username, action, details) VALUES (?, ?, ?, ?)",
            (user_id, username, action, details[:500]),
        )
    except sqlite3.Error:
        pass
