"""Robustness checks for the Day-1 primary model.
1. Repeated patient-grouped splits (10 seeds): mean and range of metrics.
2. Feature ablations: does performance depend on ED fields or on timing alone?
Splits overlap in test patients, so the spread is an approximation, not a true CI.
Prints aggregate metrics only."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "features_day1.parquet"

LABEL = "label_rule2"
CATS = ["antibiotic", "admission_type", "admission_location", "first_careunit"]
NUMS = ["hours_admit_to_icu", "hours_icu_to_abx", "ed_los_hours", "is_oral_vanc"]
SEEDS = range(10)

VARIANTS = {
    "Primary (all clinical)": (CATS, NUMS),
    "No ED features": (CATS, ["hours_admit_to_icu", "hours_icu_to_abx", "is_oral_vanc"]),
    "No ED, no admission location": (
        ["antibiotic", "admission_type", "first_careunit"],
        ["hours_admit_to_icu", "hours_icu_to_abx", "is_oral_vanc"]),
    "Timing only (ICU-to-antibiotic)": ([], ["hours_icu_to_abx"]),
}


def fit_eval(df, cats, nums, tr, te, y):
    parts = []
    if cats:
        parts.append(("cat", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                           min_frequency=50), cats))
    parts.append(("num", Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ]), nums))
    model = Pipeline([("pre", ColumnTransformer(parts)),
                      ("clf", LogisticRegression(max_iter=2000))])
    X = df[cats + nums]
    model.fit(X.iloc[tr], y[tr])
    p = model.predict_proba(X.iloc[te])[:, 1]
    return (roc_auc_score(y[te], p),
            average_precision_score(y[te], p),
            brier_score_loss(y[te], p))


def main():
    df = pd.read_parquet(DATA)
    for c in CATS:
        df[c] = df[c].fillna("missing").astype(str)
    for c in NUMS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    y = df[LABEL].astype(int).to_numpy()

    results = {name: [] for name in VARIANTS}
    for seed in SEEDS:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        tr, te = next(gss.split(df, y, groups=df["subject_id"]))
        for name, (cats, nums) in VARIANTS.items():
            results[name].append(fit_eval(df, cats, nums, tr, te, y))
        print(f"seed {seed} done", flush=True)

    print("\n== Stability across 10 patient-grouped splits ==")
    print(f"{'variant':<34}{'AUROC mean':>12}{'AUROC sd':>10}"
          f"{'AUROC min':>11}{'AUROC max':>11}{'AUPRC mean':>12}{'Brier mean':>12}")
    for name, rows in results.items():
        arr = np.array(rows)
        print(f"{name:<34}{arr[:, 0].mean():>12.3f}{arr[:, 0].std(ddof=1):>10.3f}"
              f"{arr[:, 0].min():>11.3f}{arr[:, 0].max():>11.3f}"
              f"{arr[:, 1].mean():>12.3f}{arr[:, 2].mean():>12.3f}")

    base = np.array(results["Primary (all clinical)"])[:, 0]
    print("\n== AUROC drop vs primary (paired across splits) ==")
    for name, rows in results.items():
        if name == "Primary (all clinical)":
            continue
        diff = base - np.array(rows)[:, 0]
        print(f"{name:<34} mean drop {diff.mean():+.3f}   "
              f"min {diff.min():+.3f}   max {diff.max():+.3f}")

    print("\nSplits overlap in test patients, so sd and min/max are approximate.")
    print("Use the paired drops to judge whether a feature group matters.")


if __name__ == "__main__":
    main()