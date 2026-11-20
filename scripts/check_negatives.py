import sys
from pathlib import Path

import duckdb

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import DATA_DIR, FILES

path = (DATA_DIR / FILES["microbiologyevents"]).as_posix()
con = duckdb.connect()
rows = con.execute(f"""
    SELECT
        org_name IS NULL AS no_organism,
        interpretation IS NULL AS no_interp,
        LOWER(COALESCE(comments, '')) LIKE '%no growth%' AS nogrowth_text,
        COUNT(*) AS n_rows
    FROM read_csv('{path}', all_varchar=True)
    GROUP BY 1, 2, 3
    ORDER BY n_rows DESC
""").fetchall()

print(f"{'no_organism':<12} {'no_interp':<10} {'nogrowth_text':<14} {'rows':>12}")
for r in rows:
    print(f"{str(r[0]):<12} {str(r[1]):<10} {str(r[2]):<14} {r[3]:>12,}")