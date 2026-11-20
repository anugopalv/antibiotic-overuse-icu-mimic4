"""Train Day-1 baselines (logistic regression) to predict Rule 2 overuse.
Primary model: clinical and process features only (demographics excluded on purpose).
Sensitivity model: adds race, language, marital status and insurance, to test
whether demographics change the results.
Patient-grouped train/test split. Prints aggregate metrics only.
Fitted models stay local in data/processed/."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "features_day1.parquet"
OUT = ROOT / "data" / "processed"

LABEL = "label_rule2"
CLINICAL_CATS = ["antibiotic", "admission_type", "admission_location", "first_careunit"]
CLINICAL_NUMS = ["hours_admit_to_icu", "hours_icu_to_abx", "ed_los_hours", "is_oral_vanc"]
DEMO_CATS = ["insurance", "language", "marital_status", "race"]


def build_model(cats, nums):
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist",
                              min_frequency=50), cats),
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]), nums),
    ])
    return Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=2000))])


def evaluate(name, model, X_tr, y_tr, X_te, y_te, groups_te):
    model.fit(X_tr, y_tr)
    p = model.predict_proba(X_te)[:, 1]
    thr = np.quantile(p, 0.80)
    flag = p >= thr
    bins = pd.qcut(p, 10, labels=False, duplicates="drop")
    cal = (pd.DataFrame({"bin": bins, "pred": p, "obs": y_te})
           .groupby("bin").agg(mean_pred=("pred", "mean"), observed=("obs", "mean")))
    gap = float(np.max(np.abs(cal["mean_pred"] - cal["observed"])))
    metrics = {
        "AUROC": roc_auc_score(y_te, p),
        "AUPRC": average_precision_score(y_te, p),
        "Brier": brier_score_loss(y_te, p),
        "Top20_precision": precision_score(y_te, flag),
        "Top20_recall": recall_score(y_te, flag),
        "Max_decile_calib_gap": gap,
    }
    return model, metrics


def main():
    df = pd.read_parquet(DATA)
    for c in CLINICAL_CATS + DEMO_CATS:
        df[c] = df[c].fillna("missing").astype(str)
    for c in CLINICAL_NUMS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    y = df[LABEL].astype(int).to_numpy()

    # One patient-grouped split shared by both models
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr, te = next(gss.split(df, y, groups=df["subject_id"]))

    print("== Split ==")
    print(f"Train courses: {len(tr):,}   Test courses: {len(te):,}")
    print(f"Test patients: {df['subject_id'].iloc[te].nunique():,}")
    print(f"Test prevalence (label 1): {y[te].mean() * 100:.1f}%")

    specs = {
        "Primary (clinical only)": (CLINICAL_CATS, CLINICAL_NUMS),
        "Sensitivity (with demographics)": (CLINICAL_CATS + DEMO_CATS, CLINICAL_NUMS),
    }
    results = {}
    for name, (cats, nums) in specs.items():
        X = df[cats + nums]
        model, m = evaluate(name, build_model(cats, nums),
                            X.iloc[tr], y[tr], X.iloc[te], y[te],
                            df["subject_id"].iloc[te])
        results[name] = m
        joblib.dump(model, OUT / f"day1_{'primary' if 'Primary' in name else 'sensitivity'}.joblib")

    print("\n== Test metrics (same split for both models) ==")
    keys = list(next(iter(results.values())).keys())
    print(f"{'metric':<24}" + "".join(f"{n:>34}" for n in results))
    for k in keys:
        print(f"{k:<24}" + "".join(f"{results[n][k]:>34.3f}" for n in results))

    print("\nBase rate (test): {:.3f}".format(y[te].mean()))
    print("Top20 = flag the 20% highest-risk test courses.")
    print("Max_decile_calib_gap = largest |mean predicted - observed| across deciles.")

    print("\nSaved local models (do not share): data/processed/day1_primary.joblib, "
          "day1_sensitivity.joblib")


if __name__ == "__main__":
    main()