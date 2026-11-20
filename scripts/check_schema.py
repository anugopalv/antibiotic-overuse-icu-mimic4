import sys
from pathlib import Path

import duckdb

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import DATA_DIR, FILES

con = duckdb.connect()  # in-memory, nothing written to disk

for name, fname in FILES.items():
    path = (DATA_DIR / fname).as_posix()
    # all_varchar=True avoids type-guessing errors on mixed columns
    rel = f"read_csv('{path}', all_varchar=True)"

    columns = con.execute(f"DESCRIBE SELECT * FROM {rel}").fetchall()
    n_rows = con.execute(f"SELECT COUNT(*) FROM {rel}").fetchone()[0]

    print(f"\n{name}: {n_rows:,} rows, {len(columns)} columns")
    for col_name, col_type, *_ in columns:
        print(f"  {col_name}")