"""
Historical dataset management: anonymous labelled data used only for training.
"""
from __future__ import annotations

import pandas as pd

import config
from core import database as db
from core.validation import map_columns, to_int, validate_numeric_field
from ml.trainer import load_training_data

REQUIRED = [*config.FEATURE_COLUMNS, "final_marks"]


class DatasetError(ValueError):
    """User-facing dataset error."""


def list_datasets() -> pd.DataFrame:
    return db.query_df(
        """SELECT d.id, d.name, d.filename, d.description, d.row_count, d.rejected_rows, d.include_in_training,
                  u.username AS uploaded_by, d.uploaded_at
           FROM datasets d LEFT JOIN users u ON u.id = d.uploaded_by ORDER BY d.id DESC"""
    )


def validate_historical(raw: pd.DataFrame) -> dict:
    mapping = map_columns(list(raw.columns))
    present = set(mapping.values())
    missing = [c for c in REQUIRED if c not in present]
    result = {"total": len(raw), "missing_columns": missing, "valid": pd.DataFrame(), "errors": pd.DataFrame()}
    if missing:
        return result
    df = raw.rename(columns=mapping)
    df = df.dropna(how="all")
    result["total"] = len(df)
    valid, errors = [], []
    for idx, row in enumerate(df.to_dict("records")):
        clean, problems = {}, []
        for field in REQUIRED:
            value, err = validate_numeric_field(field, row.get(field), required=True)
            clean[field] = value
            if err:
                problems.append(err)
        clean["subject"] = str(row.get("subject")).strip() if row.get("subject") not in (None, "") and \
            not pd.isna(row.get("subject")) else None
        sem = to_int(row.get("semester"))
        clean["semester"] = sem if sem and 1 <= sem <= 8 else None
        if problems:
            errors.append({"row": idx + 2, "issue": "; ".join(problems)})
        else:
            valid.append(clean)
    result["valid"] = pd.DataFrame(valid)
    result["errors"] = pd.DataFrame(errors, columns=["row", "issue"])
    return result


def save_dataset(name: str, filename: str, description: str, result: dict, acting_user: dict) -> int:
    valid = result["valid"]
    if valid.empty:
        raise DatasetError("The dataset has no valid rows.")
    name = (name or "").strip() or filename
    with db.get_db() as conn:
        cur = conn.execute(
            """INSERT INTO datasets (name, filename, description, row_count, rejected_rows, include_in_training,
                                     uploaded_by) VALUES (?, ?, ?, ?, ?, 1, ?)""",
            [name[:120], filename, (description or "")[:300], len(valid), len(result["errors"]),
             acting_user.get("id")],
        )
        did = cur.lastrowid
        cols = ["subject", "semester", *REQUIRED]
        conn.executemany(
            f"INSERT INTO historical_records (dataset_id, {', '.join(cols)}) VALUES (?, {', '.join('?' * len(cols))})",
            [[did, *[None if pd.isna(v) else v for v in row]] for row in valid[cols].astype(object).values.tolist()],
        )
    db.log_activity(acting_user.get("id"), acting_user.get("username"), "DATASET_UPLOADED",
                    f"#{did} {name} ({len(valid)} rows)")
    return did


def set_included(dataset_id: int, include: bool, acting_user: dict) -> None:
    db.execute("UPDATE datasets SET include_in_training = ? WHERE id = ?", [int(include), dataset_id])
    db.log_activity(acting_user.get("id"), acting_user.get("username"),
                    "DATASET_INCLUDED" if include else "DATASET_EXCLUDED", str(dataset_id))


def delete_dataset(dataset_id: int, acting_user: dict) -> None:
    db.execute("DELETE FROM datasets WHERE id = ?", [dataset_id])
    db.log_activity(acting_user.get("id"), acting_user.get("username"), "DATASET_DELETED", str(dataset_id))


def export_dataset(dataset_id: int) -> bytes:
    df = db.query_df(
        f"SELECT subject, semester, {', '.join(REQUIRED)} FROM historical_records WHERE dataset_id = ?",
        [dataset_id],
    )
    return df.to_csv(index=False).encode("utf-8")


def export_training_data() -> bytes:
    data, _ = load_training_data(include_historical=True)
    return data.to_csv(index=False).encode("utf-8")
