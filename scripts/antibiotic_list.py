import re
import sys
from pathlib import Path

import duckdb

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import DATA_DIR, FILES

# Generic name fragments for systemic antibiotics (IV or oral)
SYSTEMIC_TERMS = [
    "vancomycin", "cefepime", "ceftriaxone", "cefazolin", "cefpodoxime",
    "cefadroxil", "cefuroxime", "cefoxitin", "ceftazidime", "ceftaroline",
    "ceftolozane", "cefiderocol", "piperacillin", "ampicillin", "sulbactam",
    "amoxicillin", "clavulanat", "nafcillin", "oxacillin", "dicloxacillin",
    "penicillin", "ciprofloxacin", "levofloxacin", "moxifloxacin",
    "ofloxacin", "gatifloxacin", "metronidazole", "azithromycin",
    "clarithromycin", "erythromycin", "meropenem", "ertapenem",
    "imipenem", "doripenem", "aztreonam", "clindamycin", "doxycycline",
    "minocycline", "tetracycline", "tigecycline", "omadacycline",
    "daptomycin", "linezolid", "gentamicin", "tobramycin", "amikacin",
    "streptomycin", "nitrofurantoin", "fosfomycin", "sulfamethoxazole",
    "bactrim", "rifampin", "colistin",
]

# Route, formulation, or order-type fragments to exclude
EXCLUDE_TERMS = [
    "ophth", "otic", "cream", "gel", "oint", "enema", "lock",
    "inhal", "intraventricular", "intrathecal", "vaginal", "topical",
    "desensitization", "graded challenge", "test dose",
]

def normalize(name: str) -> str:
    name = name.lower().strip()
    name = re.sub(r"^\*nf\*\s*", "", name)  # strip "*nf*" prefix
    return name

def is_systemic(name: str) -> bool:
    n = normalize(name)
    if any(term in n for term in EXCLUDE_TERMS):
        return False
    return any(term in n for term in SYSTEMIC_TERMS)

if __name__ == "__main__":
    path = (DATA_DIR / FILES["prescriptions"]).as_posix()
    con = duckdb.connect()

    # Distinct drug names only, with row counts. No patient or date columns.
    rows = con.execute(f"""
        SELECT drug, COUNT(*) AS n
        FROM read_csv('{path}', all_varchar=True)
        WHERE drug IS NOT NULL
        GROUP BY drug
    """).fetchall()

    included = [(d, n) for d, n in rows if is_systemic(d)]
    total_rows = sum(n for _, n in included)

    print(f"Distinct drug names matched: {len(included):,}")
    print(f"Prescription rows matched:   {total_rows:,}")
    print("\nNames excluded that contain a systemic term (for review):")
    for d, n in sorted(rows, key=lambda x: -x[1]):
        if any(t in normalize(d) for t in SYSTEMIC_TERMS) and not is_systemic(d):
            print(f"  {d}  ({n:,})")