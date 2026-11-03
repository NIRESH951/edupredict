"""Dataset management (admin only): historical training datasets and data exports."""
from __future__ import annotations

import streamlit as st

import config
from core.synthetic import generate_historical_dataset
from services import academic_service as acs
from services import dataset_service as ds
from ui.components import empty_state, page_header, show_error, table
from ui.session import require_roles, scope


def render() -> None:
    user = require_roles("admin")
    page_header("Datasets", "Historical training datasets, sample data and full exports", "Admin")
    tabs = st.tabs(["Historical datasets", "Upload dataset", "Export & samples", "Reset demo data"])

    with tabs[0]:
        df = ds.list_datasets()
        if df.empty:
            empty_state("No historical datasets uploaded yet.")
        else:
            show = df.assign(include_in_training=df["include_in_training"].map({1: "✅ yes", 0: "no"}))
            table(show, {"id": "ID", "name": "Name", "filename": "File", "description": "Description",
                         "row_count": "Rows", "rejected_rows": "Rejected", "include_in_training": "Used for training",
                         "uploaded_by": "Uploaded by", "uploaded_at": "Uploaded"})
            opts = {r.id: f"#{r.id} · {r.name}" for r in df.itertuples()}
            c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
            did = c1.selectbox("Dataset", list(opts), format_func=lambda k: opts[k], key="ds_pick")
            row = df[df["id"] == did].iloc[0]
            c2.write("")
            c3.write("")
            c4.write("")
            if c2.button("Exclude from training" if row["include_in_training"] else "Include in training",
                         key="ds_toggle"):
                ds.set_included(did, not bool(row["include_in_training"]), user)
                st.rerun()
            c3.download_button("Download", ds.export_dataset(did), f"dataset_{did}.csv", "text/csv", key="ds_dl")
            if c4.button("Delete", key="ds_del"):
                ds.delete_dataset(did, user)
                st.success("Dataset deleted. Retrain the models to apply the change.")
                st.rerun()
            st.caption("Changes take effect the next time the models are trained.")

    with tabs[1]:
        _upload(user)

    with tabs[3]:
        _reset(user)

    with tabs[2]:
        sc = scope()
        st.markdown("##### Full exports")
        recs = acs.list_records(sc)
        st.download_button("All academic records with latest predictions (CSV)", acs.export_records_csv(recs),
                           "all_academic_records.csv", "text/csv", icon=":material/download:")
        st.download_button("Current training dataset (CSV)", ds.export_training_data(), "training_dataset.csv",
                           "text/csv", icon=":material/download:")
        st.markdown("##### Sample files")
        st.download_button("Academic records CSV template", acs.template_csv(sc), "academic_records_template.csv",
                           "text/csv", icon=":material/download:")
        st.download_button("Sample historical dataset (600 synthetic rows)",
                           generate_historical_dataset(seed=11).to_csv(index=False).encode(),
                           "sample_historical_dataset.csv", "text/csv", icon=":material/download:")


def _upload(user: dict) -> None:
    st.markdown("Upload anonymous **historical** records (past batches with known final marks) to give the "
                "models more training data. Student IDs are not needed. Required columns: "
                + ", ".join(f"`{c}`" for c in config.FEATURE_COLUMNS) + ", `final_marks`. "
                "Optional: `subject`, `semester`.")
    name = st.text_input("Dataset name", placeholder="e.g. 2024-25 batch – AIML semester 5")
    desc = st.text_input("Description (optional)")
    file = st.file_uploader("Historical dataset CSV", type=["csv"], key="ds_upload")
    if file:
        try:
            raw = acs.read_csv(file)
            result = ds.validate_historical(raw)
        except acs.RecordError as exc:
            show_error(exc)
            return
        if result["missing_columns"]:
            st.error("Missing required column(s): " + ", ".join(result["missing_columns"]))
            return
        c = st.columns(3)
        c[0].metric("Rows", result["total"])
        c[1].metric("Valid", len(result["valid"]))
        c[2].metric("Rejected", len(result["errors"]))
        if not result["errors"].empty:
            with st.expander("Rejected rows"):
                table(result["errors"], height=260)
        if not result["valid"].empty and st.button("Save dataset", type="primary", key="ds_save"):
            try:
                did = ds.save_dataset(name or file.name, file.name, desc, result, user)
                st.success(f"Dataset #{did} saved with {len(result['valid'])} rows. "
                           "Retrain the models on the Model Training page to use it.")
            except ds.DatasetError as exc:
                show_error(exc)


def _reset(user: dict) -> None:
    from core.seed import reset_demo_data
    from ui.session import logout

    st.markdown("Restore the application to its original demo state: the synthetic students, records, teachers, "
                "subjects, demo accounts and freshly trained models. Useful on a public demo link after visitors "
                "have changed data.")
    st.warning("This permanently deletes **all** current data, uploaded datasets, user accounts and model versions. "
               "Everyone, including you, will be signed out.", icon="⚠️")
    with st.form("reset_demo"):
        confirm = st.text_input("Type RESET to confirm")
        go = st.form_submit_button("Reset all data to the demo state", type="primary")
    if go:
        if confirm.strip() != "RESET":
            st.error("Please type RESET in capital letters to confirm.")
            return
        bar = st.progress(0.0, text="Resetting...")
        reset_demo_data(progress=lambda f, m: bar.progress(min(max(f, 0.0), 1.0), text=m))
        logout("Demo data was reset. Sign in again with the demo accounts.")
        st.rerun()
