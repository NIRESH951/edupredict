"""
Retrain all models from the command line and refresh stored predictions.

Usage:  python scripts/train_models.py [--no-historical]
"""
import argparse

import _bootstrap  # noqa: F401

from core import database as db
from core.scope import Scope
from ml import trainer
from services import prediction_service


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-historical", action="store_true", help="ignore uploaded historical datasets")
    args = parser.parse_args()
    db.init_db()
    result = trainer.train_models(include_historical=not args.no_historical,
                                  progress=lambda f, m: print(f"[{f:4.0%}] {m}"))
    m = result["metrics"]
    rf, lr = m["regression"]["reg_primary"]["test"], m["regression"]["reg_baseline"]["test"]
    rfc, dtc = m["classification"]["clf_primary"]["test"], m["classification"]["clf_baseline"]["test"]
    print(f"\nModel version {result['version']}  ({m['n_train']} train / {m['n_test']} test)")
    print(f"  Random Forest Regressor : MAE {rf['MAE']:.2f}  RMSE {rf['RMSE']:.2f}  R2 {rf['R2']:.3f}")
    print(f"  Linear Regression       : MAE {lr['MAE']:.2f}  RMSE {lr['RMSE']:.2f}  R2 {lr['R2']:.3f}")
    print(f"  Random Forest Classifier: Acc {rfc['Accuracy']:.3f}  F1(macro) {rfc['F1 (macro)']:.3f}")
    print(f"  Decision Tree Classifier: Acc {dtc['Accuracy']:.3f}  F1(macro) {dtc['F1 (macro)']:.3f}")
    n = prediction_service.predict_and_store(Scope.admin())
    print(f"Refreshed predictions for {n} records.")


if __name__ == "__main__":
    main()
