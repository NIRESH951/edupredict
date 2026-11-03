"""
Running the ML models over stored academic records and saving the results.
"""
from __future__ import annotations

import pandas as pd

from core import database as db
from core.scope import Scope
from ml import predictor
from ml.features import FEATURES


def _load_records(scope: Scope, record_ids: list[int] | None = None, only_missing: bool = False,
                  only_pending: bool = False) -> pd.DataFrame:
    clause, params = scope.student_filter("r.student_id")
    sql = f"SELECT r.id AS record_id, r.student_id, {', '.join('r.' + f for f in FEATURES)} " \
          f"FROM academic_records r WHERE {clause}"
    if record_ids:
        sql += f" AND r.id IN ({', '.join('?' * len(record_ids))})"
        params += [int(i) for i in record_ids]
    if only_pending:
        sql += " AND r.final_marks IS NULL"
    if only_missing:
        active = predictor.load_bundle().version
        sql += " AND r.id NOT IN (SELECT record_id FROM latest_predictions WHERE model_version = ?)"
        params.append(active)
    return db.query_df(sql, params)


def predict_and_store(scope: Scope, record_ids: list[int] | None = None, *, only_missing: bool = False,
                      only_pending: bool = False, acting_user: dict | None = None) -> int:
    """Predict for the selected records in scope and save results. Returns the number saved."""
    bundle = predictor.load_bundle()
    records = _load_records(scope, record_ids, only_missing, only_pending)
    if records.empty:
        return 0
    preds = predictor.predict_frame(records, bundle)
    user_id = acting_user.get("id") if acting_user else None
    rows = [
        [int(rid), sid, float(p.predicted_marks), float(p.baseline_marks), p.performance_category,
         p.risk_level, float(p.risk_confidence), float(p.prob_low), float(p.prob_moderate),
         float(p.prob_high), p.model_version, user_id]
        for rid, sid, p in zip(records["record_id"], records["student_id"], preds.itertuples())
    ]
    db.execute_many(
        """INSERT INTO predictions (record_id, student_id, predicted_marks, baseline_marks, performance_category,
               risk_level, risk_confidence, prob_low, prob_moderate, prob_high, model_version, predicted_by)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    if acting_user:
        db.log_activity(user_id, acting_user.get("username"), "PREDICTIONS_RUN",
                        f"{len(rows)} records with {bundle.version}")
    return len(rows)


def try_predict_records(scope: Scope, record_ids: list[int], acting_user: dict | None = None) -> int:
    """Best-effort prediction after a save; returns 0 if no model exists yet."""
    if not record_ids:
        return 0
    try:
        return predict_and_store(scope, record_ids, acting_user=acting_user)
    except predictor.ModelNotAvailable:
        return 0


def prediction_history(record_id: int) -> pd.DataFrame:
    return db.query_df(
        """SELECT created_at, model_version, ROUND(predicted_marks, 1) AS predicted_marks,
                  risk_level, ROUND(risk_confidence * 100, 1) AS confidence_pct
           FROM predictions WHERE record_id = ? ORDER BY id DESC""",
        [record_id],
    )


def prune_old_predictions(keep_per_record: int = 5) -> int:
    """Housekeeping: keep only the newest N predictions per record."""
    with db.get_db() as conn:
        cur = conn.execute(
            """DELETE FROM predictions WHERE id IN (
                   SELECT id FROM (
                       SELECT id, ROW_NUMBER() OVER (PARTITION BY record_id ORDER BY id DESC) AS rn
                       FROM predictions) WHERE rn > ?)""",
            [keep_per_record],
        )
        return cur.rowcount
