"""
Weak-subject detection and personalised study recommendations.

The recommendation engine is rule-based and transparent: every suggestion is
triggered by a specific, visible academic indicator, and the ML model is used
to estimate how much a realistic improvement could change predicted marks.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

import config
from ml import predictor
from ml.features import FEATURES
from services.analytics_service import current_records

LEVEL_ORDER = {"Critical": 0, "Weak": 1, "Needs attention": 2, "On track": 3, "Strong": 4}
PRIORITY_ORDER = {"High": 0, "Medium": 1, "Low": 2}


@dataclass
class Recommendation:
    priority: str          # High / Medium / Low
    category: str          # Attendance, Internal Assessment, ...
    title: str
    detail: str
    actions: list[str] = field(default_factory=list)
    subjects: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return self.__dict__.copy()


# --------------------------------------------------------------------------- #
# Weak subjects
# --------------------------------------------------------------------------- #
def assess_subject(row: dict) -> dict:
    """Score one subject record and list the reasons it needs attention."""
    has_actual = row.get("final_marks") is not None and not pd.isna(row.get("final_marks"))
    marks = float(row["final_marks"]) if has_actual else (
        float(row["predicted_marks"]) if row.get("predicted_marks") is not None
        and not pd.isna(row.get("predicted_marks")) else None)
    source = "actual" if has_actual else "predicted"
    internal_pct = float(row["internal1"]) + float(row["internal2"])
    assignment_pct = float(row["assignment"]) / 20 * 100
    issues: list[str] = []
    severity = 0.0

    if marks is not None:
        if marks < config.MODERATE_RISK_MIN:
            issues.append(f"{source.title()} marks {marks:.0f}/100 (High Risk band)")
            severity += 3
        elif marks < config.WEAK_MARKS_THRESHOLD:
            issues.append(f"{source.title()} marks {marks:.0f}/100 (below {config.WEAK_MARKS_THRESHOLD:.0f})")
            severity += 2
    if internal_pct < 40:
        issues.append(f"Very low internal test average ({internal_pct:.0f}%)")
        severity += 2
    elif internal_pct < config.WEAK_INTERNAL_PCT_THRESHOLD:
        issues.append(f"Internal test average {internal_pct:.0f}%")
        severity += 1
    if float(row["attendance"]) < config.ATTENDANCE_MINIMUM:
        issues.append(f"Attendance {row['attendance']:.0f}% (minimum {config.ATTENDANCE_MINIMUM:.0f}%)")
        severity += 2
    if assignment_pct < config.WEAK_ASSIGNMENT_PCT_THRESHOLD:
        issues.append(f"Assignment score {row['assignment']:g}/20")
        severity += 1
    if float(row["internal1"]) - float(row["internal2"]) >= config.INTERNAL_DROP_ALERT:
        issues.append(f"Dropped from {row['internal1']:g} to {row['internal2']:g} between tests")
        severity += 1
    if float(row["study_hours"]) < config.LOW_STUDY_HOURS and severity > 0:
        issues.append(f"Only {row['study_hours']:g} study hours/week")
        severity += 0.5

    if (marks is not None and marks < config.MODERATE_RISK_MIN) or severity >= 4:
        level = "Critical"
    elif (marks is not None and marks < config.WEAK_MARKS_THRESHOLD) or severity >= 2:
        level = "Weak"
    elif severity >= 1:
        level = "Needs attention"
    elif marks is not None and marks < config.LOW_RISK_MIN:
        level = "On track"
    else:
        level = "Strong"
    return {
        "record_id": row.get("id"),
        "subject": row["subject"],
        "semester": int(row["semester"]),
        "marks": round(marks, 1) if marks is not None else None,
        "marks_source": source,
        "attendance": float(row["attendance"]),
        "internal_pct": round(internal_pct, 1),
        "assignment": float(row["assignment"]),
        "study_hours": float(row["study_hours"]),
        "risk_level": row.get("risk_level"),
        "severity": severity,
        "level": level,
        "issues": issues,
    }


def subject_assessment(records: pd.DataFrame) -> pd.DataFrame:
    """Assessment of the student's current-semester subjects, weakest first."""
    cur = current_records(records)
    if cur.empty:
        return pd.DataFrame()
    rows = [assess_subject(r) for r in cur.to_dict("records")]
    df = pd.DataFrame(rows)
    df["_order"] = df["level"].map(LEVEL_ORDER)
    return df.sort_values(["_order", "marks"], na_position="last").drop(columns="_order").reset_index(drop=True)


def weak_subjects(records: pd.DataFrame) -> pd.DataFrame:
    df = subject_assessment(records)
    return df[df["level"].isin(["Critical", "Weak"])] if not df.empty else df


def past_difficulties(records: pd.DataFrame) -> pd.DataFrame:
    """Completed subjects (final marks known) with results below the weak threshold."""
    if records.empty:
        return records
    done = records.dropna(subset=["final_marks"])
    done = done[done["final_marks"] < config.WEAK_MARKS_THRESHOLD]
    return done[["semester", "subject", "final_marks", "attendance", "internal_avg_pct"]].sort_values(
        ["semester", "final_marks"])


# --------------------------------------------------------------------------- #
# What-if improvement estimates
# --------------------------------------------------------------------------- #
def improvement_targets(row: dict) -> dict:
    """Realistic changes a student can still make before the final exam."""
    return {
        "attendance": max(float(row["attendance"]), config.TARGET_ATTENDANCE),
        "study_hours": max(float(row["study_hours"]), config.TARGET_STUDY_HOURS),
        "assignment": max(float(row["assignment"]), 16.0),
    }


def improvement_scenarios(assessment: pd.DataFrame, records: pd.DataFrame) -> pd.DataFrame:
    """For weak subjects awaiting the exam, estimate the gain from realistic improvements."""
    if assessment.empty or not predictor.models_available():
        return pd.DataFrame()
    bundle = predictor.load_bundle()
    targets = assessment[(assessment["level"] != "Strong") & (assessment["marks_source"] == "predicted")]
    rows = []
    by_id = records.set_index("id")
    for rec in targets.itertuples():
        base = by_id.loc[rec.record_id]
        feats = {f: float(base[f]) for f in FEATURES}
        changes = improvement_targets(feats)
        if all(abs(changes[k] - feats[k]) < 1e-9 for k in changes):
            continue
        result = predictor.what_if(feats, changes, bundle)
        changed = []
        if changes["attendance"] > feats["attendance"]:
            changed.append(f"attendance {feats['attendance']:.0f}% → {changes['attendance']:.0f}%")
        if changes["study_hours"] > feats["study_hours"]:
            changed.append(f"study {feats['study_hours']:g} → {changes['study_hours']:g} hrs/week")
        if changes["assignment"] > feats["assignment"]:
            changed.append(f"assignment {feats['assignment']:g} → {changes['assignment']:g}/20")
        rows.append({
            "subject": rec.subject,
            "current_prediction": result["before"]["predicted_marks"],
            "improved_prediction": result["after"]["predicted_marks"],
            "gain": result["gain"],
            "current_risk": result["before"]["risk_level"],
            "improved_risk": result["after"]["risk_level"],
            "changes": "; ".join(changed),
        })
    return pd.DataFrame(rows).sort_values("gain", ascending=False) if rows else pd.DataFrame()


# --------------------------------------------------------------------------- #
# Study plan
# --------------------------------------------------------------------------- #
def study_plan(assessment: pd.DataFrame) -> pd.DataFrame:
    if assessment.empty:
        return pd.DataFrame()
    rows = []
    for r in assessment.itertuples():
        current = r.study_hours
        if r.level == "Critical":
            target = max(current + 4, 10)
        elif r.level == "Weak":
            target = max(current + 2, config.TARGET_STUDY_HOURS)
        elif r.level in ("Needs attention", "On track"):
            target = max(current + 1, 7)
        else:
            target = max(current, 5)
        focus = []
        if r.internal_pct < config.WEAK_INTERNAL_PCT_THRESHOLD:
            focus.append("re-learn concepts from internal test syllabus")
        if r.attendance < config.ATTENDANCE_MINIMUM:
            focus.append("cover missed lectures")
        if r.assignment < 12:
            focus.append("complete assignment practice problems")
        if not focus:
            focus.append("advanced problems and revision" if r.level == "Strong" else "previous-year question papers")
        rows.append({
            "subject": r.subject,
            "status": r.level,
            "current_hours": round(current, 1),
            "recommended_hours": round(target, 1),
            "extra_hours": round(target - current, 1),
            "focus": "; ".join(focus).capitalize(),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Recommendations
# --------------------------------------------------------------------------- #
def _names(df: pd.DataFrame, limit: int = 4) -> str:
    names = df["subject"].tolist()
    return ", ".join(names[:limit]) + (f" and {len(names) - limit} more" if len(names) > limit else "")


def generate_recommendations(records: pd.DataFrame, overall_risk: str | None = None) -> list[Recommendation]:
    """Personalised recommendations for one student's records."""
    assessment = subject_assessment(records)
    recs: list[Recommendation] = []
    if assessment.empty:
        return [Recommendation("Low", "Getting started", "No academic records yet",
                               "Recommendations will appear once your teacher adds your academic records.")]

    low_att = assessment[assessment["attendance"] < config.ATTENDANCE_MINIMUM]
    if not low_att.empty:
        recs.append(Recommendation(
            "High", "Attendance",
            f"Attendance below {config.ATTENDANCE_MINIMUM:.0f}% in {len(low_att)} subject(s)",
            f"{_names(low_att)} {'is' if len(low_att) == 1 else 'are'} below the eligibility threshold. "
            "Low attendance is one of the strongest predictors of poor final results in this model.",
            [f"Attend every remaining class of {_names(low_att)}",
             f"Target at least {config.TARGET_ATTENDANCE:.0f}% attendance for the rest of the semester",
             "Collect notes for missed lectures from classmates or the course portal within the same week",
             "Speak to your subject teacher about any make-up or tutorial sessions"],
            low_att["subject"].tolist()))
    else:
        mid = assessment[assessment["attendance"] < config.TARGET_ATTENDANCE]
        if not mid.empty:
            recs.append(Recommendation(
                "Medium", "Attendance", "Push attendance above 85%",
                f"You meet the minimum, but {_names(mid)} {'is' if len(mid) == 1 else 'are'} below "
                f"{config.TARGET_ATTENDANCE:.0f}%. Higher attendance is linked with higher final marks.",
                ["Avoid missing classes in these subjects for the rest of the term",
                 "Sit in the front rows and ask at least one question per week"],
                mid["subject"].tolist()))

    weak_int = assessment[assessment["internal_pct"] < 50]
    if not weak_int.empty:
        recs.append(Recommendation(
            "High", "Internal Assessment", f"Low internal test scores in {len(weak_int)} subject(s)",
            f"Your internal test average is below 50% in {_names(weak_int)}. Internal test marks carry the "
            "largest weight in the prediction model.",
            ["Re-attempt every question you lost marks on in Test 1 and Test 2, without notes",
             "List the topics behind each wrong answer and revise them first",
             "Attend remedial or tutorial classes offered for these subjects",
             "Solve at least two previous-year question papers per subject under timed conditions"],
            weak_int["subject"].tolist()))

    dropping = assessment[assessment.apply(
        lambda r: any("Dropped from" in i for i in r["issues"]), axis=1)]
    if not dropping.empty:
        recs.append(Recommendation(
            "Medium", "Internal Assessment", "Marks dropped between internal tests",
            f"Your Test 2 score was lower than Test 1 in {_names(dropping)}. The topics taught after Test 1 "
            "may need extra attention.",
            ["Review the syllabus covered between Test 1 and Test 2 first",
             "Ask your teacher which concepts most students found difficult",
             "Form a small study group to discuss these units"],
            dropping["subject"].tolist()))

    weak_asg = assessment[assessment["assignment"] < 12]
    if not weak_asg.empty:
        recs.append(Recommendation(
            "Medium", "Assignments", f"Improve assignment scores in {len(weak_asg)} subject(s)",
            f"Assignment scores in {_names(weak_asg)} are below 60%. Assignments are the easiest marks to "
            "recover before the exam.",
            ["Submit every remaining assignment before the deadline",
             "Ask for feedback on your last submission and apply it",
             "Use the evaluation rubric as a checklist before submitting"],
            weak_asg["subject"].tolist()))

    needs_time = assessment[(assessment["level"].isin(["Critical", "Weak"]))
                            & (assessment["study_hours"] < config.TARGET_STUDY_HOURS)]
    if not needs_time.empty:
        recs.append(Recommendation(
            "High" if (needs_time["level"] == "Critical").any() else "Medium",
            "Study Time", "Increase focused study time for weak subjects",
            f"You study fewer than {config.TARGET_STUDY_HOURS:.0f} hours per week in {_names(needs_time)}. "
            "See the weekly study plan for subject-wise targets.",
            ["Schedule fixed daily study slots and protect them",
             "Use 25-minute focused sessions with short breaks (Pomodoro technique)",
             "Use active recall: close the book and write down what you remember",
             "Revise each topic again after 1 day, 1 week and 3 weeks (spaced repetition)"],
            needs_time["subject"].tolist()))

    cur = current_records(records)
    gpa = float(cur["previous_gpa"].iloc[0]) if not cur.empty else None
    if gpa is not None and gpa < 6.0:
        past = past_difficulties(records)
        detail = f"Your previous GPA is {gpa:.2f}/10."
        if not past.empty:
            detail += f" Earlier subjects with low results: {_names(past)}."
        recs.append(Recommendation(
            "Medium", "Foundations", "Strengthen fundamentals",
            detail + " Gaps in earlier subjects often make current subjects harder.",
            ["Meet your faculty mentor to plan a foundation-revision schedule",
             "Revise prerequisite topics for this semester's weakest subject first",
             "Use free online lectures for the concepts you find hardest"]))

    if overall_risk == config.RISK_HIGH:
        recs.insert(0, Recommendation(
            "High", "Academic Support", "Meet your faculty mentor this week",
            "Your overall outlook is High Risk. Early support makes a large difference before the final exam.",
            ["Book a meeting with your mentor or class teacher this week",
             "Agree on weekly check-ins until the final examination",
             "Join a peer study group for your weakest subjects",
             "Ask about counselling support if stress or personal issues are affecting your studies"]))

    strong = assessment[assessment["level"] == "Strong"]
    if not strong.empty:
        recs.append(Recommendation(
            "Low", "Strengths", "Keep up the momentum in your strong subjects",
            f"You are doing well in {_names(strong)}.",
            ["Maintain your current routine in these subjects",
             "Attempt harder problems or a small project to deepen understanding",
             "Help classmates - explaining concepts strengthens your own understanding"],
            strong["subject"].tolist()))

    if (assessment["marks_source"] == "predicted").any():
        recs.append(Recommendation(
            "Low" if overall_risk == config.RISK_LOW else "Medium", "Exam Preparation",
            "Final examination preparation plan",
            "Start structured revision at least three weeks before the final examination.",
            ["Week 1: finish revising all units, weakest subject first",
             "Week 2: solve previous-year papers under exam timing",
             "Week 3: revise formulas, definitions and your error log; sleep well before exams"]))

    recs.sort(key=lambda r: PRIORITY_ORDER[r.priority])
    return recs


def teacher_actions(risk_row: dict) -> list[str]:
    """Suggested interventions for a teacher, given a student's risk summary row."""
    actions = []
    level = risk_row.get("overall_risk")
    if level == config.RISK_HIGH:
        actions += ["Schedule a one-to-one mentoring meeting this week",
                    "Enrol the student in remedial classes for the weakest subjects",
                    "Inform the mentor/parents and agree on a weekly progress check"]
    elif level == config.RISK_MODERATE:
        actions += ["Check in with the student after the next class",
                    "Share targeted practice material for the weakest subject"]
    if risk_row.get("avg_attendance", 100) < config.ATTENDANCE_MINIMUM:
        actions.append("Monitor attendance closely and issue an attendance warning if needed")
    if risk_row.get("avg_internal_pct", 100) < 50:
        actions.append("Offer a re-test or extra tutorial on internal-test topics")
    if not actions:
        actions.append("No intervention needed - encourage the student to keep up the good work")
    return actions
