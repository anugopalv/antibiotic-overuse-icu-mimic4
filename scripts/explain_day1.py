"""SHAP explanations for the primary Day-1 model (clinical features only).
Prints aggregate importance only. Saves a summary plot to outputs/ (no patient rows)."""
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # save to file, no window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "features_day1.parquet"
MODEL = ROOT / "data" / "processed" / "day1_primary.joblib"
OUTDIR = ROOT / "outputs"

LABEL = "label_rule2"
CATS = ["antibiotic", "admission_type", "admission_location", "first_careunit"]
NUMS = ["hours_admit_to_icu", "hours_icu_to_abx", "ed_los_hours", "is_oral_vanc"]
SAMPLE = 3000


def main():
    df = pd.read_parquet(DATA)
    for c in CATS:
        df[c] = df[c].fillna("missing").astype(str)
    for c in NUMS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    y = df[LABEL].astype(int).to_numpy()

    # Same split as train_day1_model.py (same seed and grouping)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    _, te = next(gss.split(df, y, groups=df["subject_id"]))

    model = joblib.load(MODEL)
    X_te = df[CATS + NUMS].iloc[te]
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_te), size=min(SAMPLE, len(X_te)), replace=False)
    X_s = X_te.iloc[idx]

    pre = model.named_steps["pre"]
    clf = model.named_steps["clf"]
    X_t = pre.transform(X_s)
    names = pre.get_feature_names_out()
    if hasattr(X_t, "toarray"):
        X_t = X_t.toarray()

    # Linear model: exact SHAP values, using the training-data mean as background
    explainer = shap.LinearExplainer(clf, X_t, feature_perturbation="interventional")
    sv = explainer.shap_values(X_t)

    print(f"Explained test courses: {len(X_s):,}")
    print("\n== Mean |SHAP| by feature (global importance) ==")
    imp = pd.Series(np.abs(sv).mean(axis=0), index=names).sort_values(ascending=False)
    for name, v in imp.head(15).items():
        print(f"  {name:<50} {v:.4f}")

    # Grouped importance: one line per original variable, so one-hot levels roll up
    print("\n== Mean |SHAP| by original variable ==")
    grouped = {}
    for name, v in zip(names, np.abs(sv).mean(axis=0)):
        base = name.split("__", 1)[1]
        for c in CATS:
            if base.startswith(c + "_"):
                base = c
                break
        grouped[base] = grouped.get(base, 0) + v
    for name, v in sorted(grouped.items(), key=lambda x: -x[1]):
        print(f"  {name:<50} {v:.4f}")

    # Summary plot: aggregate, contains no patient rows
    OUTDIR.mkdir(exist_ok=True)
    shap.summary_plot(sv, X_t, feature_names=names, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(OUTDIR / "shap_summary_primary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\nSaved plot: outputs/shap_summary_primary.png (aggregate, safe to share)")


if __name__ == "__main__":
    main()