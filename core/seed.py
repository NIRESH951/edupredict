"""
First-run database seeding: subjects, synthetic students and records, teachers,
class assignments, a sample historical dataset, demo login accounts, trained
models and initial predictions.

Run automatically by app.py when the database is empty, or manually with
`python scripts/reset_database.py`.
"""
from __future__ import annotations

import config
from core import database as db
from core.synthetic import generate_cohort, generate_historical_dataset, subject_catalogue

DEMO_PASSWORDS = {"admin": "Admin@123", "teacher": "Teacher@123", "student": "Student@123"}

TEACHERS = [
    # teacher_id, name, email, department, designation, phone, [(semester, section), ...]
    ("TCH001", "Dr. Ananya Sharma", "ananya.sharma@edupredict.edu", "Computer Science (AIML)", "Professor",
     "+91 98450 10001", [(5, "A"), (5, "B"), (7, "A")]),
    ("TCH002", "Prof. Rajesh Rao", "rajesh.rao@edupredict.edu", "Computer Science (AIML)", "Associate Professor",
     "+91 98450 10002", [(3, "A"), (3, "B"), (7, "B")]),
    ("TCH003", "Dr. Kavitha Iyer", "kavitha.iyer@edupredict.edu", "Computer Science", "Professor",
     "+91 98450 10003", [(3, "A"), (3, "B"), (5, "A"), (5, "B")]),
    ("TCH004", "Prof. Suresh Kulkarni", "suresh.kulkarni@edupredict.edu", "Computer Science",
     "Assistant Professor", "+91 98450 10004", [(7, "A"), (7, "B")]),
    ("TCH005", "Dr. Meena Nair", "meena.nair@edupredict.edu", "Information Science", "Associate Professor",
     "+91 98450 10005", [(3, "A"), (3, "B"), (5, "A")]),
    ("TCH006", "Prof. Arvind Hegde", "arvind.hegde@edupredict.edu", "Information Science",
     "Assistant Professor", "+91 98450 10006", [(5, "B"), (7, "A"), (7, "B")]),
]


RESET_TABLES = [
    "activity_log", "predictions", "model_registry", "historical_records", "datasets", "academic_records",
    "users", "teacher_assignments", "teachers", "subjects", "students",
]


def reset_demo_data(progress=None) -> dict:
    """
    Wipe every table and trained model, then re-create the original demo data.
    Used by the admin "Reset demo data" action on public demo deployments.
    """
    import shutil

    from ml import predictor

    with db.get_db() as conn:
        conn.execute("PRAGMA foreign_keys = OFF")
        for table in RESET_TABLES:
            conn.execute(f"DELETE FROM {table}")
        conn.execute("DELETE FROM sqlite_sequence")
        conn.execute("PRAGMA foreign_keys = ON")
    for folder in config.MODEL_DIR.glob("v*"):
        shutil.rmtree(folder, ignore_errors=True)
    predictor.clear_cache()
    return seed_database(progress=progress)


def is_seeded() -> bool:
    db.init_db()
    return db.table_count("users") > 0


def seed_database(progress=None, train: bool = True) -> dict:
    """Populate an empty database. Returns a summary. Does nothing if users already exist."""
    # Imported here to avoid circular imports (services import core.database).
    from services import analytics_service, auth_service, prediction_service
    from ml import trainer
    from core.scope import Scope

    def report(frac: float, msg: str) -> None:
        if progress:
            progress(frac, msg)

    db.init_db()
    if db.table_count("users") > 0:
        return {"seeded": False}

    report(0.05, "Creating subject catalogue...")
    subjects = subject_catalogue()
    db.execute_many(
        "INSERT INTO subjects (subject_code, subject_name, department, semester, credits) VALUES (?, ?, ?, ?, ?)",
        [[s["subject_code"], s["subject_name"], s["department"], s["semester"], s["credits"]] for s in subjects],
    )

    report(0.12, "Generating synthetic students and academic records...")
    students, records = generate_cohort()
    db.execute_many(
        """INSERT INTO students (student_id, reg_no, full_name, email, department, semester, academic_year,
                                 section, enrollment_date) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        students[["student_id", "reg_no", "full_name", "email", "department", "semester", "academic_year",
                  "section", "enrollment_date"]].values.tolist(),
    )
    rec_cols = ["student_id", "subject", "semester", *config.FEATURE_COLUMNS, "final_marks"]
    rec_rows = [[None if (c == "final_marks" and v != v) else v for c, v in zip(rec_cols, row)]
                for row in records[rec_cols].astype(object).values.tolist()]
    db.execute_many(
        f"INSERT INTO academic_records ({', '.join(rec_cols)}) VALUES ({', '.join('?' * len(rec_cols))})",
        rec_rows,
    )

    report(0.2, "Creating teachers and class assignments...")
    for tid, name, email, dept, desig, phone, classes in TEACHERS:
        db.execute(
            "INSERT INTO teachers (teacher_id, full_name, email, department, designation, phone) VALUES (?, ?, ?, ?, ?, ?)",
            [tid, name, email, dept, desig, phone],
        )
        for sem, sec in classes:
            db.execute(
                "INSERT INTO teacher_assignments (teacher_id, department, semester, section) VALUES (?, ?, ?, ?)",
                [tid, dept, sem, sec],
            )

    report(0.25, "Creating demo accounts...")
    admin_id = auth_service.create_user("admin", DEMO_PASSWORDS["admin"], "admin", "System Administrator",
                                        "admin@edupredict.edu")
    for i, (tid, name, email, *_rest) in enumerate(TEACHERS, start=1):
        auth_service.create_user(f"teacher{i}", DEMO_PASSWORDS["teacher"], "teacher", name, email, teacher_id=tid)

    report(0.3, "Adding a sample historical dataset...")
    hist = generate_historical_dataset()
    ds_id = db.execute(
        """INSERT INTO datasets (name, filename, description, row_count, rejected_rows, include_in_training, uploaded_by)
           VALUES (?, ?, ?, ?, 0, 1, ?)""",
        ["Sample historical dataset (2023-24 batch)", "sample_historical_dataset.csv",
         "Anonymous synthetic records from a previous batch, used only for model training.", len(hist), admin_id],
    )
    hist_cols = ["subject", "semester", *config.FEATURE_COLUMNS, "final_marks"]
    db.execute_many(
        f"INSERT INTO historical_records (dataset_id, {', '.join(hist_cols)}) VALUES (?, {', '.join('?' * len(hist_cols))})",
        [[ds_id, *row] for row in hist[hist_cols].values.tolist()],
    )

    summary = {"seeded": True, "students": len(students), "records": len(records), "historical": len(hist)}
    admin_user = auth_service.public_user(auth_service.get_user(admin_id))

    if train:
        report(0.35, "Training machine-learning models (first run only)...")
        result = trainer.train_models(
            acting_user=admin_user,
            progress=lambda f, m: report(0.35 + 0.5 * f, m),
        )
        summary["model_version"] = result["version"]
        report(0.88, "Generating predictions for all records...")
        summary["predictions"] = prediction_service.predict_and_store(Scope.admin(), acting_user=admin_user)

    report(0.95, "Linking demo student accounts...")
    _create_demo_students(train)
    db.log_activity(admin_id, "system", "DATABASE_SEEDED", str(summary))
    report(1.0, "Ready!")
    return summary


def _create_demo_students(have_predictions: bool) -> None:
    """Pick three students from teacher1's (Dr. Sharma's) classes with different risk levels."""
    from core.scope import Scope
    from services import analytics_service, auth_service

    pool = db.query_df(
        """SELECT s.student_id, s.full_name, s.email FROM students s
           JOIN teacher_assignments ta ON s.department = ta.department AND s.semester = ta.semester
                AND s.section = ta.section
           WHERE ta.teacher_id = 'TCH001' ORDER BY s.semester, s.section, s.student_id"""
    )
    chosen: list[str] = []
    if have_predictions:
        risk = analytics_service.student_risk_table(Scope.admin())
        risk = risk[risk["student_id"].isin(pool["student_id"])]
        # Prefer representative (not extreme) students: closest to a typical mark for each band.
        targets = {config.RISK_MODERATE: 60.0, config.RISK_HIGH: 45.0, config.RISK_LOW: 78.0}
        for level, target in targets.items():
            candidates = risk[(risk["overall_risk"] == level) & (~risk["student_id"].isin(chosen))]
            if not candidates.empty:
                best = (candidates["avg_predicted"] - target).abs().idxmin()
                chosen.append(candidates.loc[best, "student_id"])
    for sid in pool["student_id"]:
        if len(chosen) >= 3:
            break
        if sid not in chosen:
            chosen.append(sid)

    for i, sid in enumerate(chosen, start=1):
        row = pool[pool["student_id"] == sid].iloc[0]
        auth_service.create_user(f"student{i}", DEMO_PASSWORDS["student"], "student", row["full_name"],
                                 row["email"], student_id=sid)
