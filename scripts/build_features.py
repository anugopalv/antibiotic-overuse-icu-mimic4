"""Build the Day-1 feature table for ICU antibiotic courses.
Uses only admissions and icustays, plus the course table from build_courses.py.
Excluded: age (patients table not used), icustays.los (known only at discharge),
hospital_expire_flag (outcome leakage), and all culture fields (label source).
Writes a patient-level file to data/processed/ (local only). Prints counts only."""
import sys
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from config import DATA_DIR, FILES

COURSES = ROOT / "data" / "processed" / "course_rules.parquet"


def csv(key):
    return f"read_csv('{(DATA_DIR / FILES[key]).as_posix()}', all_varchar=True)"


def main():
    if not COURSES.exists():
        sys.exit("Missing data/processed/course_rules.parquet. Run build_courses.py first.")

    con = duckdb.connect()
    con.execute(f"CREATE VIEW courses AS SELECT * FROM read_parquet('{COURSES.as_posix()}')")

    con.execute(f"""
        CREATE TABLE adm AS
        SELECT hadm_id,
               TRY_CAST(admittime AS TIMESTAMP) AS admittime,
               TRY_CAST(edregtime AS TIMESTAMP) AS edregtime,
               TRY_CAST(edouttime AS TIMESTAMP) AS edouttime,
               admission_type, admission_location, insurance,
               language, marital_status, race
        FROM {csv('admissions')}
    """)
    con.execute(f"""
        CREATE TABLE icu AS
        SELECT hadm_id, stay_id, first_careunit,
               TRY_CAST(intime AS TIMESTAMP) AS icu_in
        FROM {csv('icustays')}
    """)

    # Day-1 population: course starts within 24h of an ICU admission.
    # Strict denominator: only admissions that have at least one culture.
    # The label uses timely_culture_ts, so it is never a feature.
    con.execute("""
        CREATE TABLE day1 AS
        SELECT c.subject_id, c.hadm_id, c.course_id, c.generic, c.is_oral,
               c.course_start, c.timely_culture_ts,
               i.stay_id, i.first_careunit, i.icu_in,
               a.admittime, a.edregtime, a.edouttime,
               a.admission_type, a.admission_location, a.insurance,
               a.language, a.marital_status, a.race
        FROM courses c
        JOIN icu i
          ON i.hadm_id = c.hadm_id
         AND c.course_start BETWEEN i.icu_in AND i.icu_in + INTERVAL 24 HOUR
        JOIN adm a ON a.hadm_id = c.hadm_id
        WHERE NOT c.no_culture_adm
        QUALIFY ROW_NUMBER() OVER (PARTITION BY c.course_id ORDER BY i.icu_in) = 1
    """)

    con.execute("""
        CREATE TABLE features AS
        SELECT subject_id, hadm_id, course_id,
               CAST(timely_culture_ts IS NULL AS INT) AS label_rule2,
               generic AS antibiotic,
               CAST(is_oral AS INT) AS is_oral_vanc,
               admission_type, admission_location, insurance,
               language, marital_status, race, first_careunit,
               date_diff('minute', admittime, icu_in) / 60.0 AS hours_admit_to_icu,
               date_diff('minute', icu_in, course_start) / 60.0 AS hours_icu_to_abx,
               CASE WHEN edregtime IS NOT NULL AND edouttime IS NOT NULL
                    THEN date_diff('minute', edregtime, edouttime) / 60.0
               END AS ed_los_hours
        FROM day1
    """)

    def q(sql):
        return con.execute(sql).fetchone()

    n, pats, adms, pos = q("""
        SELECT COUNT(*), COUNT(DISTINCT subject_id), COUNT(DISTINCT hadm_id),
               SUM(label_rule2) FROM features""")
    n, pats, adms, pos = n or 0, pats or 0, adms or 0, pos or 0
    print("== Day-1 feature table ==")
    print(f"Courses starting within 24h of ICU admission: {n:,}")
    print(f"Patients: {pats:,}   Admissions: {adms:,}")
    print(f"Label 1 (no timely culture): {pos:,} ({100 * pos / n:.1f}%)" if n else "No rows")
    print(f"Label 0 (timely culture): {n - pos:,}")

    print("\n== Missing values per feature ==")
    for col in ["admission_type", "insurance", "race", "first_careunit",
                "hours_admit_to_icu", "hours_icu_to_abx", "ed_los_hours"]:
        (miss,) = q(f"SELECT SUM(CAST({col} IS NULL AS INT)) FROM features")
        print(f"  {col}: {miss or 0:,}")

    out = ROOT / "data" / "processed"
    out.mkdir(parents=True, exist_ok=True)
    target = (out / "features_day1.parquet").as_posix()
    con.execute(f"COPY features TO '{target}' (FORMAT PARQUET)")
    print("\nSaved local, patient-level file (do not share): data/processed/features_day1.parquet")


if __name__ == "__main__":
    main()