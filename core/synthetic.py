"""
Synthetic academic data generator.

Produces a realistic (but entirely fictional) cohort: students across three
departments and three running semesters, with two semesters of completed
history (final marks known) and a current semester awaiting the final exam.

Final marks are generated from a hidden, non-linear function of the six
pre-exam indicators plus noise, so the ML models have a real signal to learn
but cannot fit the data perfectly.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

import config

COMMON_SUBJECTS = {
    1: [("Engineering Mathematics I", 2), ("Engineering Physics", 1), ("Basic Electrical Engineering", 0),
        ("Programming in C", -1), ("Engineering Graphics", -2)],
    2: [("Engineering Mathematics II", 3), ("Engineering Chemistry", 0), ("Basic Electronics", 1),
        ("Python Programming", -3), ("Elements of Mechanical Engineering", 0)],
}

DEPT_SUBJECTS = {
    "Computer Science (AIML)": {
        3: [("Mathematics for AI", 3), ("Data Structures", 2), ("Digital Logic Design", 0),
            ("Object Oriented Programming with Java", -1), ("Computer Organization", 1)],
        4: [("Design and Analysis of Algorithms", 4), ("Database Management Systems", -1), ("Operating Systems", 1),
            ("Probability and Statistics", 2), ("Introduction to Artificial Intelligence", -2)],
        5: [("Machine Learning", 2), ("Computer Networks", 1), ("Software Engineering", -3),
            ("Theory of Computation", 5), ("Data Visualization", -3)],
        6: [("Deep Learning", 3), ("Natural Language Processing", 2), ("Cloud Computing", -2),
            ("Compiler Design", 4), ("Big Data Analytics", 0)],
        7: [("Computer Vision", 2), ("Reinforcement Learning", 4), ("Generative AI", 0),
            ("Information Security", 1), ("MLOps", -2)],
    },
    "Computer Science": {
        3: [("Discrete Mathematics", 3), ("Data Structures", 2), ("Digital Logic Design", 0),
            ("Object Oriented Programming with Java", -1), ("Computer Organization", 1)],
        4: [("Design and Analysis of Algorithms", 4), ("Database Management Systems", -1), ("Operating Systems", 1),
            ("Microcontrollers", 2), ("Graph Theory", 3)],
        5: [("Computer Networks", 1), ("Software Engineering", -3), ("Theory of Computation", 5),
            ("Web Technologies", -2), ("Artificial Intelligence", 1)],
        6: [("Compiler Design", 4), ("Cloud Computing", -2), ("Machine Learning", 2),
            ("Cryptography and Network Security", 3), ("Mobile Application Development", -2)],
        7: [("Distributed Systems", 3), ("Internet of Things", 0), ("Big Data Analytics", 0),
            ("Blockchain Technology", 1), ("DevOps", -2)],
    },
    "Information Science": {
        3: [("Discrete Mathematics", 3), ("Data Structures", 2), ("Analog and Digital Electronics", 1),
            ("Object Oriented Programming with Java", -1), ("Unix System Programming", 0)],
        4: [("Design and Analysis of Algorithms", 4), ("Database Management Systems", -1), ("Operating Systems", 1),
            ("Software Testing", -2), ("Linear Algebra", 2)],
        5: [("Computer Networks", 1), ("Information Systems Management", -3), ("Theory of Computation", 5),
            ("Web Technologies", -2), ("Data Mining", 1)],
        6: [("Cloud Computing", -2), ("Machine Learning", 2), ("Information Retrieval", 1),
            ("Cyber Security", 2), ("Full Stack Development", -1)],
        7: [("Distributed Systems", 3), ("Internet of Things", 0), ("Big Data Analytics", 0),
            ("Information Security", 1), ("Project Management", -3)],
    },
}

FIRST_NAMES = [
    "Aarav", "Aditi", "Akash", "Ananya", "Arjun", "Bhavana", "Chetan", "Deepa", "Dhruv", "Divya", "Gaurav",
    "Harini", "Ishaan", "Jahnavi", "Karthik", "Kavya", "Lakshmi", "Manoj", "Meera", "Mohit", "Nandini", "Neha",
    "Nikhil", "Pooja", "Pranav", "Priya", "Rahul", "Riya", "Rohan", "Sahana", "Sanjay", "Shreya", "Siddharth",
    "Sneha", "Suhas", "Tanvi", "Tejas", "Varun", "Vidya", "Vikram", "Yash", "Zara", "Abhishek", "Anjali",
    "Darshan", "Farhan", "Keerthi", "Mehul", "Nisha", "Omkar", "Pallavi", "Rakesh", "Sakshi", "Tarun",
]
LAST_NAMES = [
    "Sharma", "Rao", "Iyer", "Patel", "Reddy", "Nair", "Kulkarni", "Gowda", "Shetty", "Joshi", "Menon",
    "Hegde", "Bhat", "Desai", "Kumar", "Singh", "Verma", "Gupta", "Pillai", "Naik", "Kamath", "Shenoy",
    "Mehta", "Agarwal", "Chatterjee", "Das", "Khan", "Fernandes", "Mishra", "Pandey",
]


def subject_catalogue() -> list[dict]:
    """All subjects with codes, e.g. AI501 = AIML semester 5 subject 1."""
    rows = []
    for dept in config.DEPARTMENTS:
        code = config.DEPARTMENT_CODES[dept]
        for sem in range(1, 8):
            subjects = COMMON_SUBJECTS[sem] if sem in COMMON_SUBJECTS else DEPT_SUBJECTS[dept][sem]
            for i, (name, _difficulty) in enumerate(subjects, start=1):
                rows.append({
                    "subject_code": f"{code}{sem}{i:02d}",
                    "subject_name": name,
                    "department": dept,
                    "semester": sem,
                    "credits": 3 if i == 5 else 4,
                })
    return rows


def _difficulty(dept: str, sem: int, subject: str) -> float:
    subjects = COMMON_SUBJECTS[sem] if sem in COMMON_SUBJECTS else DEPT_SUBJECTS[dept][sem]
    return float(dict(subjects)[subject])


def final_marks_function(attendance, internal1, internal2, assignment, study_hours, previous_gpa,
                         difficulty, noise):
    """Hidden data-generating process (non-linear, with thresholds and a ceiling effect)."""
    internal_total = internal1 + internal2  # out of 100
    marks = (
        3.0
        + 0.40 * internal_total
        + 0.13 * attendance
        + 0.70 * assignment
        + 7.0 * np.log1p(study_hours)                          # diminishing returns on study time
        + 2.4 * (previous_gpa - 6.0)
        - 0.60 * np.clip(75 - attendance, 0, None)             # extra penalty below 75 % attendance
        - 0.35 * np.clip(45 - internal_total, 0, None)         # weak internals compound
        + 0.05 * np.clip(internal2 - internal1, -15, 15)       # improving students do slightly better
        - 0.9 * difficulty
        + noise
    )
    marks = np.where(marks > 80, 80 + (marks - 80) * 0.6, marks)  # high scores are harder to earn
    marks = np.where(marks < 30, 30 - (30 - marks) * 0.45, marks)  # very low scores are rarer in practice
    return np.clip(marks, 0, 100)


def generate_cohort(students_per_section: int = 8, seed: int = config.RANDOM_STATE,
                    academic_year: str = "2026-27") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (students_df, records_df).
    Current semesters are 3, 5 and 7 (odd term); each student has records for the
    two previous semesters (with final marks) and the current one (final marks pending).
    """
    rng = np.random.default_rng(seed)
    start_year = int(academic_year[:4])
    students, records = [], []
    used_names: set[str] = set()
    counter = 0

    for dept in config.DEPARTMENTS:
        dcode = config.DEPARTMENT_CODES[dept]
        for current_sem in (3, 5, 7):
            batch_year = start_year - (current_sem - 1) // 2
            for section in ("A", "B"):
                for _ in range(students_per_section):
                    counter += 1
                    student_id = f"STU{counter:04d}"
                    while True:
                        name = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
                        if name not in used_names:
                            used_names.add(name)
                            break
                    roll = sum(1 for s in students if s["department"] == dept and s["semester"] == current_sem) + 1
                    reg_no = f"1EP{batch_year % 100:02d}{dcode}{roll:03d}"
                    students.append({
                        "student_id": student_id,
                        "reg_no": reg_no,
                        "full_name": name,
                        "email": f"{name.lower().replace(' ', '.')}.{reg_no.lower()}@student.edupredict.edu",
                        "department": dept,
                        "semester": current_sem,
                        "academic_year": academic_year,
                        "section": section,
                        "enrollment_date": date(batch_year, 8, int(rng.integers(1, 20))).isoformat(),
                    })

                    ability = rng.normal(0, 1)
                    diligence = 0.45 * ability + np.sqrt(1 - 0.45 ** 2) * rng.normal(0, 1)
                    gpa = float(np.clip(7.0 + 1.15 * ability + 0.3 * diligence + rng.normal(0, 0.35), 4.0, 9.9))

                    for sem in (current_sem - 2, current_sem - 1, current_sem):
                        subjects = COMMON_SUBJECTS[sem] if sem in COMMON_SUBJECTS else DEPT_SUBJECTS[dept][sem]
                        drift_a = ability + rng.normal(0, 0.3)
                        drift_d = diligence + rng.normal(0, 0.35)
                        sem_marks = []
                        for subject, _ in subjects:
                            diff = _difficulty(dept, sem, subject)
                            att = np.clip(80 + 8.5 * drift_d + 2.5 * drift_a + rng.normal(0, 5.5), 38, 100)
                            sh = np.clip(5.2 + 2.1 * drift_d + 0.7 * drift_a + rng.normal(0, 1.6), 0.5, 18)
                            i1 = np.clip(30.5 + 7.0 * drift_a + 1.8 * drift_d - 0.6 * diff + rng.normal(0, 4.2), 6, 50)
                            i2 = np.clip(i1 + 1.6 * drift_d + rng.normal(0.5, 3.8), 6, 50)
                            asg = np.clip(14.0 + 2.4 * drift_d + 0.9 * drift_a + rng.normal(0, 1.8), 2, 20)
                            final = final_marks_function(att, i1, i2, asg, sh, gpa, diff, rng.normal(0, 6.5))
                            is_current = sem == current_sem
                            records.append({
                                "student_id": student_id,
                                "subject": subject,
                                "semester": sem,
                                "attendance": round(float(att), 1),
                                "internal1": round(float(i1) * 2) / 2,
                                "internal2": round(float(i2) * 2) / 2,
                                "assignment": round(float(asg) * 2) / 2,
                                "study_hours": round(float(sh), 1),
                                "previous_gpa": round(gpa, 2),
                                "final_marks": None if is_current else round(float(final), 1),
                            })
                            sem_marks.append(final)
                        # GPA for the next semester follows this semester's results.
                        gpa = float(np.clip(0.55 * gpa + 0.45 * (np.mean(sem_marks) / 10 + 0.4)
                                            + rng.normal(0, 0.15), 4.0, 9.9))

    return pd.DataFrame(students), pd.DataFrame(records)


def generate_historical_dataset(n_rows: int = 600, seed: int = 7) -> pd.DataFrame:
    """Anonymous labelled dataset in the CSV upload format (for the Datasets module demo)."""
    rng = np.random.default_rng(seed)
    rows = []
    all_subjects = [(d, s, name, diff) for d in config.DEPARTMENTS
                    for s, subs in {**COMMON_SUBJECTS, **DEPT_SUBJECTS[d]}.items() for name, diff in subs]
    for _ in range(n_rows):
        dept, sem, subject, diff = all_subjects[int(rng.integers(0, len(all_subjects)))]
        a, d = rng.normal(0, 1), rng.normal(0, 1)
        d = 0.45 * a + np.sqrt(1 - 0.45 ** 2) * d
        gpa = float(np.clip(7.0 + 1.15 * a + 0.3 * d + rng.normal(0, 0.35), 4.0, 9.9))
        att = np.clip(80 + 8.5 * d + 2.5 * a + rng.normal(0, 5.5), 38, 100)
        sh = np.clip(5.2 + 2.1 * d + 0.7 * a + rng.normal(0, 1.6), 0.5, 18)
        i1 = np.clip(30.5 + 7.0 * a + 1.8 * d - 0.6 * diff + rng.normal(0, 4.2), 6, 50)
        i2 = np.clip(i1 + 1.6 * d + rng.normal(0.5, 3.8), 6, 50)
        asg = np.clip(14.0 + 2.4 * d + 0.9 * a + rng.normal(0, 1.8), 2, 20)
        final = final_marks_function(att, i1, i2, asg, sh, gpa, diff, rng.normal(0, 6.5))
        rows.append({
            "subject": subject, "semester": sem,
            "attendance": round(float(att), 1), "internal1": round(float(i1) * 2) / 2,
            "internal2": round(float(i2) * 2) / 2, "assignment": round(float(asg) * 2) / 2,
            "study_hours": round(float(sh), 1), "previous_gpa": round(gpa, 2),
            "final_marks": round(float(final), 1),
        })
    return pd.DataFrame(rows)
