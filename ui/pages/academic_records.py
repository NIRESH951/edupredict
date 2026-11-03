"""Academic data management (Module 4)."""
from __future__ import annotations

import streamlit as st

import config
from core.scope import AccessDenied
from services import academic_service as acs
from services import prediction_service, student_service, subject_service
from ui.components import (
    RISK_COLUMN_CONFIG,
    empty_state,
    fmt,
    page_header,
    risk_badge,
    show_error,
    table,
)
from ui.session import require_roles, scope

OTHER = "Other (type a subject name)"


def render() -> None:
    user = require_roles("admin", "teacher")
    sc = scope()
    page_header("Academic Records",
                "Enter, correct, search and import attendance, internal marks and exam results",
                "All students" if user["role"] == "admin" else "Your assigned students")
    tabs = st.tabs(["View & search", "Add record", "Edit / delete", "Import CSV"])
    with tabs[0]:
        _view(sc)
    with tabs[1]:
        _add(sc, user)
    with tabs[2]:
        _edit(sc, user)
    with tabs[3]:
        _import(sc, user)


def _view(sc) -> None:
    c = st.columns([2, 1.4, 0.8, 1.6, 1.4])
    search = c[0].text_input("Search", placeholder="Student, reg. no. or subject", key="rec_search")
    dept = c[1].selectbox("Department", ["All"] + config.DEPARTMENTS, key="rec_dept")
    sem = c[2].selectbox("Semester", ["All"] + config.SEMESTERS, key="rec_sem")
    subj = c[3].selectbox("Subject", ["All"] + acs.subjects_in_records(sc), key="rec_subj")
    status = c[4].selectbox("Status", ["All", "Awaiting final exam", "Final marks available"], key="rec_status")
    df = acs.list_records(sc, search, dept, sem, None, subj, None, status)
    st.caption(f"{len(df):,} record(s)")
    if df.empty:
        empty_state("No records match these filters.")
        return
    cols = ["id", "student_id", "reg_no", "full_name", "subject", "semester", "attendance", "internal1", "internal2",
            "assignment", "study_hours", "previous_gpa", "final_marks", "predicted_marks", "risk_level"]
    table(df[cols], {**RISK_COLUMN_CONFIG, "id": "Record", "student_id": "Student ID", "reg_no": "Reg. No.",
                     "full_name": "Name", "subject": "Subject", "semester": "Sem", "risk_level": "Predicted risk"},
          height=520)
    st.download_button("Download these records (CSV)", acs.export_records_csv(df[cols]), "academic_records.csv",
                       "text/csv", icon=":material/download:")


def _subject_input(prefix: str, department: str, semester: int, current: str | None = None) -> str:
    names = subject_service.subject_names(department, semester)
    options = names + [OTHER]
    index = options.index(current) if current in options else (len(options) - 1 if current else 0)
    choice = st.selectbox("Subject *", options, index=index, key=f"{prefix}_subj")
    if choice == OTHER:
        return st.text_input("Subject name *", current or "", key=f"{prefix}_subj_other")
    return choice


def _record_inputs(prefix: str, e: dict) -> dict:
    c = st.columns(3)
    vals = {
        "attendance": c[0].number_input("Attendance (%) *", 0.0, 100.0, float(e.get("attendance", 80.0)), 0.5,
                                        key=f"{prefix}_att"),
        "internal1": c[1].number_input("Internal Test 1 (/50) *", 0.0, 50.0, float(e.get("internal1", 30.0)), 0.5,
                                       key=f"{prefix}_i1"),
        "internal2": c[2].number_input("Internal Test 2 (/50) *", 0.0, 50.0, float(e.get("internal2", 30.0)), 0.5,
                                       key=f"{prefix}_i2"),
    }
    c = st.columns(3)
    vals["assignment"] = c[0].number_input("Assignment score (/20) *", 0.0, 20.0, float(e.get("assignment", 14.0)),
                                           0.5, key=f"{prefix}_asg")
    vals["study_hours"] = c[1].number_input("Weekly study hours *", 0.0, 60.0, float(e.get("study_hours", 6.0)),
                                            0.5, key=f"{prefix}_sh",
                                            help="Self-study hours per week for this subject")
    vals["previous_gpa"] = c[2].number_input("Previous GPA (/10) *", 0.0, 10.0, float(e.get("previous_gpa", 7.0)),
                                             0.01, key=f"{prefix}_gpa")
    has_final = e.get("final_marks") is not None
    c = st.columns([1, 2])
    available = c[0].checkbox("Final exam marks available", value=has_final, key=f"{prefix}_hasfinal",
                              help="Leave unticked before the final examination – the system will predict it.")
    final = c[1].number_input("Actual final examination marks (/100)", 0.0, 100.0,
                              float(e["final_marks"]) if has_final else 0.0, 0.5, key=f"{prefix}_final",
                              disabled=not available)
    vals["final_marks"] = final if available else None
    return vals


def _show_prediction(sc, record_id: int, student_id: str) -> None:
    rec = acs.list_records(sc, student_id=student_id)
    row = rec[rec["id"] == record_id]
    if row.empty or row["predicted_marks"].isna().all():
        st.info("Saved. No trained model is active yet, so no prediction was made.")
        return
    r = row.iloc[0]
    st.markdown(f"**Prediction:** {fmt(r['predicted_marks'])} / 100 · {r['performance_category']} · "
                f"{risk_badge(r['risk_level'])}", unsafe_allow_html=True)


def _add(sc, user) -> None:
    options = student_service.student_options(sc)
    if not options:
        empty_state("No students available. Add students first.")
        return
    sid = st.selectbox("Student *", list(options), format_func=lambda k: options[k], key="add_rec_student")
    student = student_service.get_student(sid, sc)
    c = st.columns([1, 3])
    semester = c[0].selectbox("Semester *", list(range(1, int(student["semester"]) + 1)),
                              index=int(student["semester"]) - 1, key="add_rec_sem",
                              help="Records can be added for the current or any earlier semester.")
    with c[1]:
        subject = _subject_input("add_rec", student["department"], semester)
    vals = _record_inputs("add_rec", {})
    if st.button("Save record", type="primary", key="add_rec_save"):
        try:
            rid = acs.add_record({"student_id": sid, "subject": subject, "semester": semester, **vals}, sc, user)
            prediction_service.try_predict_records(sc, [rid], user)
            st.success(f"Record #{rid} saved for {student['full_name']} – {subject}.")
            _show_prediction(sc, rid, sid)
        except (acs.RecordError, AccessDenied) as exc:
            show_error(exc)


def _edit(sc, user) -> None:
    options = student_service.student_options(sc)
    if not options:
        empty_state("No students available.")
        return
    sid = st.selectbox("Student", list(options), format_func=lambda k: options[k], key="edit_rec_student")
    recs = acs.list_records(sc, student_id=sid)
    if recs.empty:
        empty_state("This student has no academic records yet.")
        return
    labels = {r.id: f"Sem {r.semester} · {r.subject}" + (" · final entered" if r.final_marks == r.final_marks else "")
              for r in recs.itertuples()}
    rid = st.selectbox("Record", list(labels), format_func=lambda k: labels[k], key="edit_rec_pick")
    rec = acs.get_record(rid, sc)
    student = student_service.get_student(sid, sc)
    prefix = f"edit_{rid}"
    c = st.columns([1, 3])
    semester = c[0].selectbox("Semester", list(range(1, int(student["semester"]) + 1)),
                              index=min(int(rec["semester"]), int(student["semester"])) - 1, key=f"{prefix}_sem")
    with c[1]:
        subject = _subject_input(prefix, student["department"], semester, rec["subject"])
    vals = _record_inputs(prefix, rec)
    b1, b2 = st.columns([1, 3])
    if b1.button("Save changes", type="primary", key=f"{prefix}_save"):
        try:
            changed = acs.update_record(rid, {"subject": subject, "semester": semester, **vals}, sc, user)
            if changed:
                prediction_service.try_predict_records(sc, [rid], user)
                st.success("Record updated and prediction refreshed (input values changed).")
            else:
                st.success("Record updated.")
            _show_prediction(sc, rid, sid)
        except (acs.RecordError, AccessDenied) as exc:
            show_error(exc)

    st.divider()
    with st.form(f"{prefix}_delete"):
        sure = st.checkbox(f"Delete this record ({labels[rid]}) and its predictions")
        go = st.form_submit_button("Delete record")
    if go:
        if not sure:
            st.error("Please tick the confirmation box.")
        else:
            try:
                acs.delete_record(rid, sc, user)
                st.success("Record deleted.")
                st.rerun()
            except AccessDenied as exc:
                show_error(exc)


def _import(sc, user) -> None:
    st.markdown("Upload a CSV with one row per student per subject. Required columns: "
                + ", ".join(f"`{c}`" for c in config.ACADEMIC_CSV_REQUIRED)
                + ". `final_marks` is optional (leave blank before the exam).")
    st.download_button("Download CSV template", acs.template_csv(sc), "academic_records_template.csv", "text/csv",
                       icon=":material/download:")
    with st.expander("Column rules"):
        rules = "\n".join(
            f"- **{c}** – {config.FEATURE_LABELS.get(c, c)}"
            + (f", {config.FIELD_LIMITS[c][0]:g} to {config.FIELD_LIMITS[c][1]:g}" if c in config.FIELD_LIMITS else "")
            for c in config.ACADEMIC_CSV_COLUMNS)
        st.markdown(rules + "\n\nCommon header variations such as *Internal Test 1*, *Attendance Percentage* or "
                    "*Previous GPA* are recognised automatically. The student ID must already exist"
                    + (" and belong to one of your classes." if sc.is_teacher else "."))

    file = st.file_uploader("Academic records CSV", type=["csv"], key="rec_upload")
    if not file:
        return
    try:
        raw = acs.read_csv(file)
    except acs.RecordError as exc:
        show_error(exc)
        return
    result = acs.validate_academic_csv(raw, sc)
    if not result.structure_ok:
        st.error("Missing required column(s): " + ", ".join(result.missing_columns))
        st.caption("Columns found: " + ", ".join(map(str, raw.columns)))
        return

    valid_n = len(result.valid)
    c = st.columns(5)
    c[0].metric("Rows in file", result.total_rows)
    c[1].metric("Valid rows", valid_n)
    c[2].metric("Rows with errors", result.error_rows)
    c[3].metric("Duplicates in file", result.duplicates_in_file)
    c[4].metric("Already in database", result.existing_in_db)
    for w in result.warnings:
        st.warning(w)

    missing = result.missing_values[result.missing_values["missing"] > 0]
    if not missing.empty:
        with st.expander(f"Missing values in {len(missing)} column(s)", expanded=True):
            table(missing, {"column": "Column", "missing": "Blank cells", "required": "Required"})
    if not result.errors.empty:
        with st.expander(f"Upload errors ({len(result.errors)})", expanded=True):
            table(result.errors, {"row": "CSV row", "student_id": "Student ID", "issue": "Problem"},
                  height=min(320, 36 * len(result.errors) + 40))
            st.download_button("Download error report", result.errors.to_csv(index=False).encode(),
                               "upload_errors.csv", "text/csv")
    if valid_n == 0:
        st.error("No valid rows to import. Please fix the errors and upload again.")
        return
    with st.expander("Preview valid rows"):
        table(result.valid.drop(columns=["exists"]).head(50))

    mode = "skip"
    if result.existing_in_db:
        mode = st.radio(f"{result.existing_in_db} row(s) match existing records (same student, subject and semester).",
                        ["skip", "update"], horizontal=True,
                        format_func=lambda m: "Skip existing records" if m == "skip" else "Update existing records")
    if st.button(f"Import {valid_n} valid row(s)", type="primary", key="do_import"):
        with st.spinner("Importing and predicting..."):
            counts = acs.import_academic_records(result.valid, sc, mode, user)
            predicted = prediction_service.try_predict_records(sc, counts["record_ids"], user)
        st.success(f"Import complete: {counts['inserted']} inserted, {counts['updated']} updated, "
                   f"{counts['skipped']} skipped, {result.error_rows} rejected. "
                   f"Predictions generated for {predicted} record(s).")
