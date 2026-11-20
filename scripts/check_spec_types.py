import sys
from pathlib import Path

import duckdb

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import DATA_DIR, FILES

path = (DATA_DIR / FILES["microbiologyevents"]).as_posix()
con = duckdb.connect()
rows = con.execute(f"""
    WITH spec AS (
        SELECT subject_id, hadm_id, micro_specimen_id,
               ANY_VALUE(spec_type_desc) AS spec_type,
               COUNT(org_name) AS n_org,
               MAX(CASE WHEN LOWER(COALESCE(comments, '')) LIKE '%no growth%'
                        THEN 1 ELSE 0 END) AS ng
        FROM read_csv('{path}', all_varchar=True)
        WHERE hadm_id IS NOT NULL
        GROUP BY subject_id, hadm_id, micro_specimen_id
    )
    SELECT spec_type,
           COUNT(*) AS specimens,
           SUM(CASE WHEN n_org = 0 AND ng = 1 THEN 1 ELSE 0 END) AS true_negatives
    FROM spec
    GROUP BY spec_type
    ORDER BY true_negatives DESC
    LIMIT 25
""").fetchall()

print(f"{'specimen type':<40} {'specimens':>12} {'true negatives':>16}")
for spec_type, n, neg in rows:
    print(f"{str(spec_type):<40} {n:>12,} {neg:>16,}")