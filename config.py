"""
Central configuration for EduPredict.

Every tunable constant (paths, academic rules, ML settings, security settings)
lives here so the rest of the code never hard-codes them.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths (can be overridden with environment variables, e.g. for tests)
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("EDUPREDICT_DATA_DIR", BASE_DIR / "data"))
MODEL_DIR = Path(os.environ.get("EDUPREDICT_MODEL_DIR", BASE_DIR / "models"))
DB_PATH = Path(os.environ.get("EDUPREDICT_DB", DATA_DIR / "edupredict.db"))

DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Application identity
# --------------------------------------------------------------------------- #
APP_NAME = "EduPredict"
APP_TAGLINE = "AI-Based Student Performance Prediction and Academic Improvement System"
INSTITUTION_NAME = "EduPredict Institute of Technology"
APP_VERSION = "1.0.0"

# --------------------------------------------------------------------------- #
# Academic structure
# --------------------------------------------------------------------------- #
DEPARTMENTS = [
    "Computer Science (AIML)",
    "Computer Science",
    "Information Science",
]
DEPARTMENT_CODES = {
    "Computer Science (AIML)": "AI",
    "Computer Science": "CS",
    "Information Science": "IS",
}
SECTIONS = ["A", "B", "C"]
SEMESTERS = list(range(1, 9))
DESIGNATIONS = ["Professor", "Associate Professor", "Assistant Professor", "Lecturer"]

# Minimum attendance required to sit the final examination.
ATTENDANCE_MINIMUM = 75.0

# --------------------------------------------------------------------------- #
# Academic record fields and their valid ranges
# --------------------------------------------------------------------------- #
# The six features that are known BEFORE the final examination.
FEATURE_COLUMNS = [
    "attendance",
    "internal1",
    "internal2",
    "assignment",
    "study_hours",
    "previous_gpa",
]
TARGET_COLUMN = "final_marks"

FEATURE_LABELS = {
    "attendance": "Attendance (%)",
    "internal1": "Internal Test 1 (/50)",
    "internal2": "Internal Test 2 (/50)",
    "assignment": "Assignment (/20)",
    "study_hours": "Weekly Study Hours",
    "previous_gpa": "Previous GPA (/10)",
    "final_marks": "Final Exam Marks (/100)",
}
SHORT_LABELS = {
    "attendance": "Attendance",
    "internal1": "Internal Test 1",
    "internal2": "Internal Test 2",
    "assignment": "Assignment",
    "study_hours": "Study Hours",
    "previous_gpa": "Previous GPA",
}

# (min, max) inclusive limits used by validation everywhere.
FIELD_LIMITS = {
    "attendance": (0.0, 100.0),
    "internal1": (0.0, 50.0),
    "internal2": (0.0, 50.0),
    "assignment": (0.0, 20.0),
    "study_hours": (0.0, 60.0),
    "previous_gpa": (0.0, 10.0),
    "final_marks": (0.0, 100.0),
    "semester": (1, 8),
}

# Column order of the CSV template for academic records.
ACADEMIC_CSV_COLUMNS = [
    "student_id",
    "subject",
    "semester",
    "attendance",
    "internal1",
    "internal2",
    "assignment",
    "study_hours",
    "previous_gpa",
    "final_marks",
]
ACADEMIC_CSV_REQUIRED = ACADEMIC_CSV_COLUMNS[:-1]  # final_marks is optional

# Friendly header aliases accepted in uploaded CSV files.
COLUMN_ALIASES = {
    "student_id": ["studentid", "student", "stu_id", "student_code"],
    "subject": ["subject_name", "course", "course_name", "subject_code"],
    "semester": ["sem"],
    "attendance": ["attendance_percentage", "attendance_pct", "attendance_percent", "attendance_%"],
    "internal1": ["internal_test_1", "internal_1", "test1", "ia1", "internal_test1", "cie1"],
    "internal2": ["internal_test_2", "internal_2", "test2", "ia2", "internal_test2", "cie2"],
    "assignment": ["assignment_score", "assignment_marks", "assignments"],
    "study_hours": ["weekly_study_hours", "study_hours_per_week", "hours_studied"],
    "previous_gpa": ["prev_gpa", "gpa", "previous_sgpa", "sgpa", "cgpa"],
    "final_marks": ["final", "final_exam_marks", "actual_final_marks", "see", "final_examination_marks", "exam_marks"],
}

# --------------------------------------------------------------------------- #
# Risk and performance categories (derived from final marks out of 100)
# --------------------------------------------------------------------------- #
LOW_RISK_MIN = 70.0       # >= 70  -> Low Risk
MODERATE_RISK_MIN = 50.0  # 50-70  -> Moderate Risk, < 50 -> High Risk

RISK_LOW = "Low Risk"
RISK_MODERATE = "Moderate Risk"
RISK_HIGH = "High Risk"
RISK_LEVELS = [RISK_LOW, RISK_MODERATE, RISK_HIGH]

# Status colours (fixed; always paired with a text label, never colour alone).
RISK_COLORS = {RISK_LOW: "#0ca30c", RISK_MODERATE: "#fab219", RISK_HIGH: "#d03b3b"}
RISK_ICONS = {RISK_LOW: "🟢", RISK_MODERATE: "🟠", RISK_HIGH: "🔴"}

# (lower bound inclusive, label) checked from top to bottom.
PERFORMANCE_BANDS = [
    (85.0, "Excellent"),
    (70.0, "Very Good"),
    (60.0, "Good"),
    (50.0, "Average"),
    (0.0, "Needs Improvement"),
]

# --------------------------------------------------------------------------- #
# Machine learning settings
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
MIN_TRAINING_SAMPLES = 30
RF_N_ESTIMATORS = 300

# --------------------------------------------------------------------------- #
# Weak-subject and recommendation thresholds
# --------------------------------------------------------------------------- #
WEAK_MARKS_THRESHOLD = 60.0          # predicted/actual marks below this = weak
WEAK_INTERNAL_PCT_THRESHOLD = 60.0   # internal test average below 60 % = weak
WEAK_ASSIGNMENT_PCT_THRESHOLD = 60.0
LOW_STUDY_HOURS = 6.0                # per subject per week
TARGET_STUDY_HOURS = 8.0
TARGET_ATTENDANCE = 85.0
INTERNAL_DROP_ALERT = 5.0            # Test 2 lower than Test 1 by >= 5 marks

# --------------------------------------------------------------------------- #
# Security settings
# --------------------------------------------------------------------------- #
PBKDF2_ALGORITHM = "sha256"
PBKDF2_ITERATIONS = 600_000          # OWASP 2023 recommendation for PBKDF2-HMAC-SHA256
PASSWORD_MIN_LENGTH = 8
MAX_FAILED_LOGINS = 5
LOCKOUT_MINUTES = 5
SESSION_TIMEOUT_MINUTES = 30

ROLES = ["admin", "teacher", "student"]
ROLE_LABELS = {"admin": "Administrator", "teacher": "Teacher", "student": "Student"}
