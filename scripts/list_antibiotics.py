import sys
from pathlib import Path

import duckdb

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import DATA_DIR, FILES

path = (DATA_DIR / FILES["prescriptions"]).as_posix()

# Broad keyword list for a first pass. We review the results by hand,
# because some matches will be false positives.
KEYWORDS = [
    "cillin", "cef", "penem", "mycin", "floxacin", "cycline",
    "vancomycin", "tazobactam", "metronidazole", "sulfamethoxazole",
    "linezolid", "clindamycin", "gentamicin", "tobramycin", "amikacin",
    "aztreonam", "colistin", "daptomycin", "rifampin", "fosfomycin",
    "nitrofurantoin", "bactrim", "cipro", "levofloxacin", "moxifloxacin",
]

where = " OR ".join(f"LOWER(drug) LIKE '%{kw}%'" for kw in KEYWORDS)

con = duckdb.connect()
rows = con.execute(f"""
    SELECT LOWER(drug) AS drug_name, COUNT(*) AS n
    FROM read_csv('{path}', all_varchar=True)
    WHERE drug IS NOT NULL AND ({where})
    GROUP BY drug_name
    ORDER BY n DESC
    LIMIT 150
""").fetchall()

print(f"{'drug name':<50} {'rows':>10}")
for name, n in rows:
    print(f"{name:<50} {n:>10,}")