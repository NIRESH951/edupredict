"""
Write the sample CSV files in data/:
  * academic_records_template.csv  - upload template
  * sample_academic_records.csv    - the synthetic cohort's records in upload format
  * sample_students.csv            - the synthetic cohort's students
  * sample_historical_dataset.csv  - anonymous labelled data for the Datasets module

Usage:  python scripts/generate_sample_data.py
"""
import _bootstrap  # noqa: F401

import config
from core.synthetic import generate_cohort, generate_historical_dataset
from services.academic_service import template_csv


def main() -> None:
    students, records = generate_cohort()
    out = config.DATA_DIR
    (out / "academic_records_template.csv").write_bytes(template_csv())
    records[config.ACADEMIC_CSV_COLUMNS].to_csv(out / "sample_academic_records.csv", index=False)
    students.to_csv(out / "sample_students.csv", index=False)
    generate_historical_dataset(seed=11).to_csv(out / "sample_historical_dataset.csv", index=False)
    print(f"Wrote sample files to {out}")


if __name__ == "__main__":
    main()
