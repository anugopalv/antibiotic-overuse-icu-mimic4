"""Streamlit dashboard for the ICU antibiotic overuse project.
Reads only outputs/dashboard_summary.json (aggregate values, no patient-level data)."""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "outputs" / "dashboard_summary.json"

st.set_page_config(page_title="ICU Antibiotic Overuse", layout="wide")

if not SUMMARY.exists():
    st.error("dashboard_summary.json not found. Run scripts\\export_dashboard_summary.py first.")
    st.stop()

S = json.loads(SUMMARY.read_text(encoding="utf-8"))
R = S["rules"]        # cohort, rule1, rule2, rule3, demo_sensitivity, cost_scenarios
M = S["model"]        # metrics, top20_flag, robustness, calibration, shap_grouped

st.title("Antibiotic Overuse Detection in ICU")
st.caption("MIMIC-IV v3.1, ICU antibiotic courses. Aggregate results only; "
           "patient-level data is not loaded or shown.")

tab_over, tab_rules, tab_model, tab_expl, tab_cost = st.tabs(
    ["Overview", "Rules", "Day-1 model", "Explanations", "Cost and caveats"])

# ---------- Overview ----------
with tab_over:
    c = R["cohort"]
    cols = st.columns(4)
    cols[0].metric("ICU antibiotic courses", f"{c['courses']:,}")
    cols[1].metric("Patients", f"{c['patients']:,}")
    cols[2].metric("Admissions", f"{c['admissions']:,}")
    cols[3].metric("Oral vancomycin courses", f"{c['oral_vancomycin_courses']:,}")

    cols = st.columns(3)
    cols[0].metric("Rule 1: no culture during admission",
                   f"{R['rule1']['pct']}%", help="Patient level")
    cols[1].metric("Rule 2: no culture in 24h before start",
                   f"{R['rule2']['all_courses_pct']}%", help="Course level, all courses")
    cols[2].metric("Rule 3 (preliminary): negative culture, continued >72h",
                   f"{R['rule3']['pct_of_all_courses']}%", help="Course level")

# ---------- Rules ----------
with tab_rules:
    rows = [
        {"Rule": "1. No culture during admission", "Level": "Patient",
         "Result": f"{R['rule1']['patients_flagged']:,} of {R['rule1']['patients_total']:,} "
                   f"patients ({R['rule1']['pct']}%)",
         "Definition": "Antibiotic patient with no microbiology rows in the admission"},
        {"Rule": "2. No culture in 24h before start (all courses)", "Level": "Course",
         "Result": f"{R['rule2']['all_courses_overuse']:,} courses "
                   f"({R['rule2']['all_courses_pct']}%)",
         "Definition": "No culture collected in the 24 hours before the antibiotic start"},
        {"Rule": "2. Strict (admissions with a culture only)", "Level": "Course",
         "Result": f"{R['rule2']['strict_overuse']:,} of {R['rule2']['strict_denominator']:,} "
                   f"({R['rule2']['strict_pct']}%)",
         "Definition": "Same as Rule 2, excluding admissions with no culture (already in Rule 1)"},
        {"Rule": "3. Negative culture, continued >72h (preliminary)", "Level": "Course",
         "Result": f"{R['rule3']['overuse']:,} of {R['rule3']['denominator_valid']:,} "
                   f"with negative culture ({R['rule3']['pct_of_negative_courses']}%)",
         "Definition": "Negative culture (no organism, 'no growth') and stop >72h after "
                       "collection time"},
    ]
    st.table(pd.DataFrame(rows))

    st.subheader("Rule 3 demo comparison")
    d = R["demo_sensitivity"]
    cols = st.columns(2)
    cols[0].metric("Demo notebook (small subset)", f"{d['notebook_demo_rule3_pct']}%")
    cols[1].metric("Same logic, full data", f"{d['full_data_rule3_pct']}%")
    st.caption(d["description"] + ". The notebook used a much smaller subset, so the "
               "two numbers are compared as a sensitivity check, not a validation.")

# ---------- Model ----------
with tab_model:
    st.write(f"Primary model: logistic regression on clinical Day-1 features. "
             f"Test set: {M['test_courses']:,} courses, {M['test_patients']:,} patients, "
             f"prevalence {M['test_prevalence']:.1%}.")

    cols = st.columns(4)
    cols[0].metric("AUROC", M["metrics"]["AUROC"])
    cols[1].metric("AUPRC", M["metrics"]["AUPRC"])
    cols[2].metric("Brier score", M["metrics"]["Brier"])
    cols[3].metric("Top-20% precision", M["top20_flag"]["precision"])

    rb = M["robustness"]
    st.write(f"Stability across {rb['splits']} patient-grouped splits: AUROC "
             f"{rb['auroc_mean']} (sd {rb['auroc_sd']}, range {rb['auroc_min']} to "
             f"{rb['auroc_max']}).")
    st.caption("Splits overlap in test patients, so the spread is approximate.")

    st.subheader("Calibration by risk decile")
    cal = pd.DataFrame(M["calibration"]).set_index("decile")
    st.line_chart(cal[["mean_predicted", "observed"]])
    st.dataframe(cal)

# ---------- Explanations ----------
with tab_expl:
    st.write("Mean absolute SHAP value by variable: how much each variable moves the "
             "predicted risk, on average.")
    shap_df = pd.DataFrame(M["shap_grouped"]).set_index("variable")
    st.bar_chart(shap_df["mean_abs_shap"])
    st.subheader("Interpretation")
    st.markdown(
        "- **Time from ICU admission to first antibiotic** is the strongest driver. "
        "Early starts are more likely to lack a culture in the preceding 24 hours.\n"
        "- **Admission location and ICU unit** add context on how the ICU stay began.\n"
        "- **ED timing** adds little once admission location is included (ablation: "
        "AUROC change about 0.004).\n"
        "- These describe workflow patterns, not patient severity. Age and vital signs "
        "are not in the model.")

# ---------- Cost and caveats ----------
with tab_cost:
    cs = R["cost_scenarios"]
    st.warning(cs["status"])
    st.write(f"Overuse courses (Rule 2, all courses): {cs['overuse_courses_used']:,}. "
             f"Assumed extra days per course: {cs['extra_days_per_course']}.")
    cost_df = pd.DataFrame([
        {"Scenario": k, "ICU day cost (USD)": v["icu_day_cost_usd"],
         "Total (USD)": f"{v['total_usd']:,}"} for k, v in cs["scenarios"].items()])
    st.table(cost_df)

    st.subheader("Limitations")
    st.markdown(
        "- The label is a documentation and timing pattern, not proof of unnecessary treatment.\n"
        "- Prescriptions record orders, not administrations. Durations are order-based.\n"
        "- Age, vital signs and severity scores are not included.\n"
        "- Rule 3 uses collection time, not report time, and may include surveillance cultures.\n"
        "- The 2.5 extra days per course is an assumption, not a measured value.")

    st.subheader("Notes")
    for n in S["notes"]:
        st.write(f"- {n}")