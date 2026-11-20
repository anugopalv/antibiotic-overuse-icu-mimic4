"""Export an aggregate-only summary for the Streamlit dashboard.
Reads local patient-level files in data/processed/, but writes only counts,
metrics, calibration bins, robustness stats and grouped feature importance to
outputs/dashboard_summary.json. The script checks that no IDs are in the output."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
OUTDIR = ROOT / "outputs"
OUTFILE = OUTDIR / "dashboard_summary.json"

LABEL = "label_rule2"
CATS = ["antibiotic", "admission_type", "admission_location", "first_careunit"]
NUMS = ["hours_admit_to_icu", "hours_icu_to_abx", "ed_los_hours", "is_oral_vanc"]
DEMO_EXTRA_DAYS = 2.5          # assumed average extra days per overuse course
COST_PER_ICU_DAY = {"USD 110 (low scenario)": 110, "USD 500 (high scenario)": 500}


def r(x, n=3):
    return None if x is None or pd.isna(x) else round(float(x), n)


def rules_summary():
    c = pd.read_parquet(PROC / "course_rules.parquet")
    n_courses = len(c)
    no_cult = c["no_culture_adm"].astype(bool)
    neg = c["neg_culture_ts"].notna()
    valid = c["valid_duration"].astype(bool)
    r2_all = c["timely_culture_ts"].isna().sum()
    r2_strict_den = (~no_cult).sum()
    r2_strict = c["rule2_overuse"].astype(bool).sum()
    r3_den = (neg & valid).sum()
    r3_over = c["rule3_overuse"].astype(bool).sum()
    return {
        "cohort": {
            "courses": int(n_courses),
            "patients": int(c["subject_id"].nunique()),
            "admissions": int(c["hadm_id"].nunique()),
            "invalid_duration_courses": int((~valid).sum()),
            "oral_vancomycin_courses": int(c["is_oral"].astype(bool).sum()),
        },
        "rule1": {
            "level": "patient",
            "patients_total": int(c["subject_id"].nunique()),
            "patients_flagged": int(c.loc[no_cult, "subject_id"].nunique()),
            "pct": r(100 * c.loc[no_cult, "subject_id"].nunique()
                     / c["subject_id"].nunique(), 1),
        },
        "rule2": {
            "level": "course",
            "all_courses_overuse": int(r2_all),
            "all_courses_pct": r(100 * r2_all / n_courses, 1),
            "strict_denominator": int(r2_strict_den),
            "strict_overuse": int(r2_strict),
            "strict_pct": r(100 * r2_strict / r2_strict_den, 1),
        },
        "rule3": {
            "level": "course",
            "negative_culture_courses": int(neg.sum()),
            "denominator_valid": int(r3_den),
            "overuse": int(r3_over),
            "pct_of_all_courses": r(100 * r3_over / n_courses, 1),
            "pct_of_negative_courses": r(100 * r3_over / r3_den, 1),
            "status": "preliminary; start point and surveillance cultures pending review",
        },
        "demo_sensitivity": {
            "description": "Demo logic (prescriptions + microbiologyevents only), full data",
            "notebook_demo_rule3_pct": 5.1,
            "full_data_rows": 331389,
            "full_data_rule3_pct": 7.1,
        },
        "cost_scenarios": {
            "status": "hypothetical sensitivity, not a guaranteed saving",
            "overuse_courses_used": int(r2_all),
            "extra_days_per_course": DEMO_EXTRA_DAYS,
            "scenarios": {
                label: {
                    "icu_day_cost_usd": cost,
                    "total_usd": int(round(r2_all * DEMO_EXTRA_DAYS * cost)),
                }
                for label, cost in COST_PER_ICU_DAY.items()
            },
        },
    }


def prep(df):
    for c in CATS:
        df[c] = df[c].fillna("missing").astype(str)
    for c in NUMS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def make_model():
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist",
                              min_frequency=50), CATS),
        ("num", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                          ("scale", StandardScaler())]), NUMS),
    ])
    return Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=2000))])


def model_summary(df, y):
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr, te = next(gss.split(df, y, groups=df["subject_id"]))
    X = df[CATS + NUMS]
    model = joblib.load(PROC / "day1_primary.joblib")
    p = model.predict_proba(X.iloc[te])[:, 1]
    yt = y[te]
    flag = p >= np.quantile(p, 0.80)

    bins = pd.qcut(p, 10, labels=False, duplicates="drop")
    cal = (pd.DataFrame({"bin": bins, "pred": p, "obs": yt})
           .groupby("bin").agg(n=("obs", "size"), mean_pred=("pred", "mean"),
                               observed=("obs", "mean")).reset_index())
    calibration = [{"decile": int(row.bin) + 1, "n": int(row.n),
                    "mean_predicted": r(row.mean_pred), "observed": r(row.observed)}
                   for row in cal.itertuples()]

    # Robustness: refit the primary model on 10 patient-grouped splits
    aurocs = []
    for seed in range(10):
        g = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        tr_s, te_s = next(g.split(df, y, groups=df["subject_id"]))
        m = make_model().fit(X.iloc[tr_s], y[tr_s])
        aurocs.append(roc_auc_score(y[te_s], m.predict_proba(X.iloc[te_s])[:, 1]))
    aurocs = np.array(aurocs)

    # Grouped SHAP importance on a fixed sample of test courses
    pre, clf = model.named_steps["pre"], model.named_steps["clf"]
    rng = np.random.default_rng(42)
    idx = rng.choice(len(te), size=min(3000, len(te)), replace=False)
    Xs = X.iloc[te].iloc[idx]
    Xt = pre.transform(Xs)
    if hasattr(Xt, "toarray"):
        Xt = Xt.toarray()
    names = pre.get_feature_names_out()
    sv = shap.LinearExplainer(clf, Xt, feature_perturbation="interventional").shap_values(Xt)
    grouped = {}
    for name, v in zip(names, np.abs(sv).mean(axis=0)):
        base = name.split("__", 1)[1]
        for c in CATS:
            if base.startswith(c + "_"):
                base = c
                break
        grouped[base] = grouped.get(base, 0.0) + float(v)
    shap_grouped = [{"variable": k, "mean_abs_shap": r(v, 4)}
                    for k, v in sorted(grouped.items(), key=lambda kv: -kv[1])]

    return {
        "split_seed": 42,
        "test_courses": int(len(te)),
        "test_patients": int(df["subject_id"].iloc[te].nunique()),
        "test_prevalence": r(yt.mean(), 3),
        "metrics": {
            "AUROC": r(roc_auc_score(yt, p)),
            "AUPRC": r(average_precision_score(yt, p)),
            "Brier": r(brier_score_loss(yt, p)),
        },
        "top20_flag": {
            "precision": r(precision_score(yt, flag)),
            "recall": r(recall_score(yt, flag)),
            "flagged_courses": int(flag.sum()),
        },
        "calibration": calibration,
        "robustness": {
            "splits": 10,
            "auroc_mean": r(aurocs.mean()),
            "auroc_sd": r(aurocs.std(ddof=1)),
            "auroc_min": r(aurocs.min()),
            "auroc_max": r(aurocs.max()),
        },
        "shap_grouped": shap_grouped,
    }


def main():
    df = prep(pd.read_parquet(PROC / "features_day1.parquet"))
    y = df[LABEL].astype(int).to_numpy()

    summary = {
        "notes": [
            "Locked definitions; Rule 2 is the model label.",
            "Rule 3 is preliminary and depends on the 72-hour start point and surveillance cultures.",
            "Cost figures are hypothetical sensitivity scenarios.",
            "Day-1 model uses clinical features only; demographics were excluded and tested as a sensitivity check.",
        ],
        "rules": rules_summary(),
        "model": model_summary(df, y),
    }

    OUTDIR.mkdir(exist_ok=True)
    text = json.dumps(summary, indent=2)
    for banned in ["subject_id", "hadm_id", "stay_id", "course_id"]:
        assert banned not in text, f"Patient-level field found in output: {banned}"
    OUTFILE.write_text(text, encoding="utf-8")

    print("Saved aggregate summary: outputs/dashboard_summary.json (safe to share)")
    print("Check passed: no patient-level identifiers in output.")
    print(f"Rule 1: {summary['rules']['rule1']['pct']}%   "
          f"Rule 2 (all): {summary['rules']['rule2']['all_courses_pct']}%   "
          f"Rule 3: {summary['rules']['rule3']['pct_of_all_courses']}%")
    print(f"Model AUROC: {summary['model']['metrics']['AUROC']}   "
          f"robustness mean: {summary['model']['robustness']['auroc_mean']}")


if __name__ == "__main__":
    main()