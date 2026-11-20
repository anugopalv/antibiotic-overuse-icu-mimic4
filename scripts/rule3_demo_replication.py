"""Sensitivity check: replicate the demo notebook's Rule 3 logic on MIMIC-IV.
Prints aggregate counts only. Nothing patient-level is written."""
import sys
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from config import DATA_DIR, FILES

DEMO_DRUGS = ["vancomycin", "meropenem", "piperacillin"]  # demo cell 1


def csv(key):
    return f"read_csv('{(DATA_DIR / FILES[key]).as_posix()}', all_varchar=True)"


def run(con, icu_scope, label):
    drug_filter = " OR ".join(f"lower(r.drug) LIKE '%{d}%'" for d in DEMO_DRUGS)
    icu_clause = ""
    if icu_scope:
        icu_clause = """
          AND EXISTS (SELECT 1 FROM icu i
                      WHERE i.hadm_id = r.hadm_id
                        AND TRY_CAST(r.starttime AS TIMESTAMP) BETWEEN i.intime AND i.outtime)"""

    # Demo unit: one row per prescription (no course grouping)
    con.execute(f"""
        CREATE OR REPLACE TABLE demo_rows AS
        SELECT r.hadm_id,
               TRY_CAST(r.starttime AS TIMESTAMP) AS start_ts,
               TRY_CAST(r.stoptime AS TIMESTAMP) AS stop_ts
        FROM {csv('prescriptions')} r
        WHERE ({drug_filter}) {icu_clause}
    """)

    # Demo negative: microbiology row with organism missing, or "no growth" in org_name
    con.execute(f"""
        CREATE OR REPLACE TABLE demo_neg AS
        SELECT hadm_id,
               COALESCE(TRY_CAST(charttime AS TIMESTAMP),
                        TRY_CAST(chartdate AS TIMESTAMP)) AS ct
        FROM {csv('microbiologyevents')}
        WHERE hadm_id IS NOT NULL
          AND (org_name IS NULL OR trim(org_name) = ''
               OR lower(org_name) LIKE '%no growth%')
    """)

    # Demo flags: negative inside the order window, and stop > charttime + 72h.
    # A null stop time gives NULL, which counts as not overuse (same as the demo).
    con.execute("""
        CREATE OR REPLACE TABLE demo_flags AS
        SELECT d.*,
          EXISTS (SELECT 1 FROM demo_neg n
                  WHERE n.hadm_id = d.hadm_id
                    AND n.ct >= d.start_ts AND n.ct <= d.stop_ts) AS has_neg_in_window,
          EXISTS (SELECT 1 FROM demo_neg n
                  WHERE n.hadm_id = d.hadm_id
                    AND n.ct >= d.start_ts AND n.ct <= d.stop_ts
                    AND d.stop_ts > n.ct + INTERVAL 72 HOUR) AS rule3
        FROM demo_rows d
    """)

    n, null_stop, neg, r3 = con.execute("""
        SELECT COUNT(*),
               SUM(CAST(stop_ts IS NULL AS INT)),
               SUM(CAST(has_neg_in_window AS INT)),
               SUM(CAST(rule3 AS INT))
        FROM demo_flags""").fetchone()
    n, null_stop, neg, r3 = n or 0, null_stop or 0, neg or 0, r3 or 0
    p_all = f"{100 * r3 / n:.1f}%" if n else "n/a"
    p_neg = f"{100 * r3 / neg:.1f}%" if neg else "n/a"

    print(f"== {label} ==")
    print(f"Prescription rows (demo drugs): {n:,}   missing stop: {null_stop:,}")
    print(f"Rows with a negative culture inside the order window: {neg:,}")
    print(f"Rule 3 (demo logic): {r3:,} of {n:,} rows ({p_all})   "
          f"of rows with negative in window ({p_neg})\n")


def main():
    con = duckdb.connect()
    con.execute(f"""
        CREATE TABLE icu AS
        SELECT hadm_id, TRY_CAST(intime AS TIMESTAMP) AS intime,
               TRY_CAST(outtime AS TIMESTAMP) AS outtime
        FROM {csv('icustays')}
    """)
    run(con, icu_scope=False, label="Whole hospital (demo scope)")
    run(con, icu_scope=True, label="ICU stays only (our cohort scope)")


if __name__ == "__main__":
    main()