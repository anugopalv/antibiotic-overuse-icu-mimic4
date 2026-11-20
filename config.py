from pathlib import Path

# Change only this line to switch between demo and full data
DATA_DIR = Path("data/raw")

FILES = {
    "prescriptions": "prescriptions.csv.gz",
    "microbiologyevents": "microbiologyevents.csv.gz",
    "admissions": "admissions.csv.gz",
    "icustays": "icustays.csv.gz",
}
