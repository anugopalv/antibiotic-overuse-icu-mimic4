"""Build ICU antibiotic courses and apply the three locked rules.
Prints aggregate counts only. Patient-level tables stay local in data/processed/,
which is excluded from Git. Output is deterministic: ties are broken by
pharmacy_id and poe_id, so repeated runs on the same data give identical counts."""
import sys
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "scripts"))
from config import DATA_DIR, FILES
from antibiotic_list import SYSTEMIC_TERMS, is_systemic

ALIASES = {"bactrim": "sulfamethoxazole"}


def generic(name):
    n = name.lower()
    for term in SYSTEMIC_TERMS:
        if term in n:
            return ALIASES.get(term, term)
    return None


def csv(key):
    return f"read_csv('{(DATA_DIR / FILES[key]).as_posix()}', all_varchar=True)"


def pct(a, b):
    return f"{100 * a / b:.1f}%" if b else "n/a"


def main():
    con = duckdb.connect()
    z = lambda x: x or 0

    # 1. Systemic antibiotic names mapped to generic names
    names = sorted(r[0] for r in con.execute(
        f"SELECT DISTINCT drug FROM {csv('prescriptions')} WHERE drug IS NOT NULL"
    ).fetchall())
    pairs = [(d, generic(d)) for d in names if is_systemic(d)]
    pairs = [(d, g) for d, g in pairs if g]
    con.execute("CREATE TABLE drug_map (drug VARCHAR, generic VARCHAR)")
    con.executemany("INSERT INTO drug_map VALUES (?, ?)", pairs)

    # 2. ICU stays
    con.execute(f"""
        CREATE TABLE icu AS
        SELECT subject_id, hadm_id, stay_id,
               TRY_CAST(intime AS TIMESTAMP) AS intime,
               TRY_CAST(outtime AS TIMESTAMP) AS outtime
        FROM {csv('icustays')}
    """)

    # 3. Antibiotic orders that start during an ICU stay.
    #    pharmacy_id and poe_id are kept as deterministic tie-breakers.
    con.execute(f"""
        CREATE TABLE abx_orders AS
        SELECT r.subject_id, r.hadm_id, m.generic,
               r.pharmacy_id, r.poe_id,
               TRY_CAST(r.starttime AS TIMESTAMP) AS start_ts,
               TRY_CAST(r.stoptime AS TIMESTAMP) AS stop_ts,
               lower(r.drug) LIKE '%oral%' AS is_oral
        FROM {csv('prescriptions')} r
        JOIN drug_map m ON r.drug = m.drug
        WHERE EXISTS (
            SELECT 1 FROM icu i
            WHERE i.hadm_id = r.hadm_id
              AND TRY_CAST(r.starttime AS TIMESTAMP) BETWEEN i.intime AND i.outtime
        )
    """)

    # 4. Courses: same antibiotic, same admission, new course after a >24h gap.
    #    ORDER BY includes tie-breakers so ordering is fixed across runs.
    ORDER = "start_ts, stop_ts, pharmacy_id, poe_id"
    con.execute(f"""
        CREATE TABLE orders_gap AS
        SELECT *, MAX(COALESCE(stop_ts, start_ts)) OVER w AS prev_max_stop
        FROM abx_orders
        WINDOW w AS (PARTITION BY hadm_id, generic ORDER BY {ORDER}
                     ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
    """)
    con.execute(f"""
        CREATE TABLE orders_c AS
        SELECT *,
               SUM(CASE WHEN prev_max_stop IS NULL
                         OR start_ts - prev_max_stop > INTERVAL 24 HOUR
                        THEN 1 ELSE 0 END)
                 OVER (PARTITION BY hadm_id, generic ORDER BY {ORDER}
                       ROWS UNBOUNDED PRECEDING) AS course_no
        FROM orders_gap
    """)
    con.execute("""
        CREATE TABLE courses AS
        SELECT ROW_NUMBER() OVER (ORDER BY hadm_id, generic, course_no) AS course_id,
               subject_id, hadm_id, generic, course_no,
               MIN(start_ts) AS course_start,
               MAX(COALESCE(stop_ts, start_ts)) AS course_stop,
               BOOL_OR(is_oral) AS is_oral,
               BOOL_OR(stop_ts IS NULL) AS null_stop,
               COUNT(*) AS n_orders
        FROM orders_c
        GROUP BY subject_id, hadm_id, generic, course_no
    """)

    # 5. Culture specimens: one row per specimen
    #    n_org = number of rows with an organism reported
    #    nogrowth_text = 1 if any row for the specimen says "no growth"
    con.execute(f"""
        CREATE TABLE specimens AS
        SELECT subject_id, hadm_id, micro_specimen_id,
               MIN(COALESCE(TRY_CAST(charttime AS TIMESTAMP),
                            TRY_CAST(chartdate AS TIMESTAMP))) AS collect_ts,
               COUNT(org_name) AS n_org,
               MAX(CASE WHEN lower(coalesce(comments, '')) LIKE '%no growth%'
                        THEN 1 ELSE 0 END) AS nogrowth_text
        FROM {csv('microbiologyevents')}
        WHERE hadm_id IS NOT NULL
        GROUP BY subject_id, hadm_id, micro_specimen_id
    """)

    # 6. Course-level rule inputs
    #    neg_culture_ts: earliest TRUE negative specimen (no organism AND "no growth")
    #    collected from 24h before course start onward, same admission
    con.execute("""
        CREATE TABLE course_rules AS
        SELECT c.*,
          NOT EXISTS (SELECT 1 FROM specimens s WHERE s.hadm_id = c.hadm_id) AS no_culture_adm,
          (SELECT MIN(s.collect_ts) FROM specimens s
            WHERE s.hadm_id = c.hadm_id
              AND s.collect_ts BETWEEN c.course_start - INTERVAL 24 HOUR
                                   AND c.course_start) AS timely_culture_ts,
          (SELECT MIN(s.collect_ts) FROM specimens s
            WHERE s.hadm_id = c.hadm_id
              AND s.n_org = 0
              AND s.nogrowth_text = 1
              AND s.collect_ts >= c.course_start - INTERVAL 24 HOUR) AS neg_culture_ts,
          date_diff('second', c.course_start, c.course_stop) / 86400.0 AS duration_days,
          (c.course_stop >= c.course_start) AS valid_duration
        FROM courses c
    """)
    con.execute("""
        CREATE TABLE rule_flags AS
        SELECT *,
          (NOT no_culture_adm AND timely_culture_ts IS NULL) AS rule2_overuse,
          (neg_culture_ts IS NOT NULL
             AND valid_duration
             AND course_stop > neg_culture_ts + INTERVAL 72 HOUR) AS rule3_overuse
        FROM course_rules
    """)

    def q(sql):
        return con.execute(sql).fetchone()

    print("== Cohort (ICU antibiotic courses) ==")
    orders, = q("SELECT COUNT(*) FROM abx_orders")
    print(f"ICU antibiotic orders: {z(orders):,}")
    courses, cpats, cadms, oral, nullstop = q("""
        SELECT COUNT(*), COUNT(DISTINCT subject_id), COUNT(DISTINCT hadm_id),
               SUM(CAST(is_oral AS INT)), SUM(CAST(null_stop AS INT)) FROM courses""")
    print(f"Courses: {z(courses):,}   Patients: {z(cpats):,}   Admissions: {z(cadms):,}")
    print(f"Oral vancomycin courses (sensitivity flag): {z(oral):,}")
    print(f"Courses with missing stop time: {z(nullstop):,}")

    invalid, = q("SELECT SUM(CAST(NOT valid_duration AS INT)) FROM course_rules")
    print(f"Courses with stop before start (excluded from duration and Rule 3): {z(invalid):,}")
    dmin, d25, dmed, d75, dmax, dmean = q("""
        SELECT MIN(duration_days), quantile_cont(duration_days, 0.25),
               quantile_cont(duration_days, 0.5), quantile_cont(duration_days, 0.75),
               MAX(duration_days), AVG(duration_days)
        FROM course_rules WHERE valid_duration""")
    print(f"Course duration, valid only (days): min {dmin:.2f}, p25 {d25:.2f}, "
          f"median {dmed:.2f}, p75 {d75:.2f}, max {dmax:.2f}, mean {dmean:.2f}")

    print("\n== Culture specimens (validation of definitions) ==")
    spec, null_t, pos, neg, neg_true = q("""
        SELECT COUNT(*), SUM(CAST(collect_ts IS NULL AS INT)),
               SUM(CAST(n_org > 0 AS INT)), SUM(CAST(n_org = 0 AS INT)),
               SUM(CAST(n_org = 0 AND nogrowth_text = 1 AS INT)) FROM specimens""")
    print(f"Specimens: {z(spec):,}   no collection time: {z(null_t):,}")
    print(f"Positive (organism reported): {z(pos):,}")
    print(f"Negative (no organism): {z(neg):,}   of which true negative ('no growth'): {z(neg_true):,}")

    print("\n== Rule 1: no culture during admission (patient level) ==")
    pts_total, pts_r1 = q("""
        SELECT COUNT(DISTINCT subject_id),
               COUNT(DISTINCT subject_id) FILTER (WHERE no_culture_adm)
        FROM course_rules""")
    print(f"Patients: {z(pts_total):,}   flagged: {z(pts_r1):,}   "
          f"({pct(z(pts_r1), z(pts_total))})")

    print("\n== Rule 2: culture not in 24h before antibiotic (course level) ==")
    all_c, no_cult_c, no_timely_all, r2_den, r2_over = q("""
        SELECT COUNT(*), SUM(CAST(no_culture_adm AS INT)),
               SUM(CAST(timely_culture_ts IS NULL AS INT)),
               SUM(CAST(NOT no_culture_adm AS INT)),
               SUM(CAST(rule2_overuse AS INT)) FROM rule_flags""")
    print(f"Courses with no timely culture, all courses: "
          f"{z(no_timely_all):,} of {z(all_c):,} ({pct(z(no_timely_all), z(all_c))})")
    print(f"Courses with no culture in admission (counted in Rule 1): {z(no_cult_c):,}")
    print(f"Strict Rule 2 denominator (admission has a culture): {z(r2_den):,}")
    print(f"Strict Rule 2 overuse: {z(r2_over):,} ({pct(z(r2_over), z(r2_den))})   "
          f"Good: {z(r2_den) - z(r2_over):,}")

    print("\n== Rule 3: true negative culture, antibiotic >72h after (course level) ==")
    r3_neg, r3_over, r3_den = q("""
        SELECT SUM(CAST(neg_culture_ts IS NOT NULL AS INT)),
               SUM(CAST(rule3_overuse AS INT)),
               SUM(CAST(neg_culture_ts IS NOT NULL AND valid_duration AS INT))
        FROM rule_flags""")
    print(f"Courses with a true negative index culture: {z(r3_neg):,}")
    print(f"Rule 3 denominator (negative culture, valid duration): {z(r3_den):,}")
    print(f"Rule 3 overuse: {z(r3_over):,} of {z(all_c):,} all courses "
          f"({pct(z(r3_over), z(all_c))})   "
          f"of negative-culture courses ({pct(z(r3_over), z(r3_den))})")

    out = ROOT / "data" / "processed"
    out.mkdir(parents=True, exist_ok=True)
    target = (out / "course_rules.parquet").as_posix()
    con.execute(f"COPY rule_flags TO '{target}' (FORMAT PARQUET)")
    print("\nSaved local, patient-level file (do not share): data/processed/course_rules.parquet")


if __name__ == "__main__":
    main()