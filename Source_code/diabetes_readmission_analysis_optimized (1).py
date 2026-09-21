"""Diabetes Hospital Readmission Analytics

Refactored from the supplied capstone Jupyter notebook.

Business question
-----------------
What patient, hospitalization, utilization, diagnosis, and medication
characteristics are associated with hospital readmission?

This is descriptive analytics. Results identify patterns and associations;
they do not establish causation.

Example
-------
python diabetes_readmission_analysis_optimized.py --input diabete_huggingface.csv

The script creates:
* diabetes_clean.csv
* diabetes_analytics.db
* printed KPI / Pandas / SQL summaries
* the same seven Matplotlib visualizations used in the notebook
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DEFAULT_INPUT = "diabete_huggingface.csv"
DEFAULT_CLEAN_CSV = "diabetes_clean.csv"
DEFAULT_DATABASE = "diabetes_analytics.db"
SQL_TABLE = "hospital_encounters"
MIN_DIAGNOSIS_ENCOUNTERS = 30

# Columns required by the analyses below. Checking them early produces a clear
# error instead of a harder-to-debug KeyError later in the workflow.
REQUIRED_COLUMNS = {
    "rowID",
    "readmitted",
    "gender",
    "race",
    "age",
    "admission_type_id",
    "discharge_disposition_id",
    "time_in_hospital",
    "num_medications",
    "num_lab_procedures",
    "number_inpatient",
    "number_outpatient",
    "number_emergency",
    "diag_1_desc",
    "diabetesMed",
    "change",
}


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def section(title: str) -> None:
    """Print a visible section heading when the notebook is run as a script."""
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}")


def show_table(title: str, table: pd.DataFrame | pd.Series, rows: int | None = None) -> None:
    """Print a DataFrame/Series cleanly because scripts do not auto-display objects."""
    print(f"\n{title}")
    output = table.head(rows) if rows is not None else table
    print(output.to_string())


def validate_columns(df: pd.DataFrame, required: Iterable[str] = REQUIRED_COLUMNS) -> None:
    """Verify that all columns needed for the project are available."""
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(
            "The input file is missing required column(s): " + ", ".join(missing)
        )


def grouped_readmission_summary(
    df: pd.DataFrame,
    group_cols: str | list[str],
) -> pd.DataFrame:
    """Return encounter count, readmissions, and readmission rate by group.

    This helper replaces repeated groupby/agg code from the notebook. The
    readmission rate is calculated as the mean of a binary 0/1 flag and then
    converted to a percentage.
    """
    summary = (
        df.groupby(group_cols, observed=True, dropna=False)
        .agg(
            Encounters=("rowID", "count"),
            Readmissions=("readmission_flag", "sum"),
            Readmission_Rate=("readmission_flag", "mean"),
        )
        .reset_index()
    )
    summary["Readmission_Rate"] = summary["Readmission_Rate"].mul(100)
    return summary


# -----------------------------------------------------------------------------
# 1. Data understanding
# -----------------------------------------------------------------------------
def load_data(input_path: Path) -> pd.DataFrame:
    """Load the raw encounter-level CSV and perform basic structural checks."""
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}. "
            "Place the CSV at this path or pass --input with the correct path."
        )

    df = pd.read_csv(input_path)
    validate_columns(df)
    return df


def data_quality_profile(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Profile native missing values and the dataset's special '?' marker."""
    profile = pd.DataFrame(
        {
            "Data_Type": df.dtypes.astype(str),
            "Missing_Count": df.isna().sum(),
            "Missing_Percent": df.isna().mean().mul(100),
            "Unique_Values": df.nunique(dropna=True),
        }
    ).sort_values("Missing_Percent", ascending=False)

    # The source dataset uses '?' as a second representation of missing data.
    # Count it separately before cleaning so the original data-quality issue is
    # visible rather than silently disappearing during replacement.
    question_marks = df.astype("string").eq("?").sum().sort_values(ascending=False)
    question_marks = question_marks[question_marks > 0]
    return profile, question_marks


# -----------------------------------------------------------------------------
# 2. Data preparation and feature engineering
# -----------------------------------------------------------------------------
def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean special missing values and create analysis-ready features."""
    # Work on a copy so the raw DataFrame remains available for auditing.
    clean = df.copy()
    clean.replace("?", np.nan, inplace=True)

    # The notebook treats readmitted as a binary outcome. Convert explicitly to
    # numeric and validate that only 0 and 1 remain, preventing silent misuse if
    # a different source file encodes readmission as text categories.
    clean["readmission_flag"] = pd.to_numeric(clean["readmitted"], errors="raise")
    observed_flags = set(clean["readmission_flag"].dropna().unique())
    if not observed_flags.issubset({0, 1}):
        raise ValueError(
            "'readmitted' must be binary (0/1) for this analysis. "
            f"Observed values: {sorted(observed_flags)}"
        )
    clean["readmission_flag"] = clean["readmission_flag"].astype("int8")

    # Total prior utilization combines inpatient, outpatient, and emergency
    # visits into one encounter-history measure.
    clean["prior_utilization"] = clean[
        ["number_inpatient", "number_outpatient", "number_emergency"]
    ].sum(axis=1)

    # Bins match the original notebook exactly: 0=None, 1-2=Low, 3-5=Medium,
    # and 6+=High. observed=True later avoids unused categorical combinations.
    clean["utilization_group"] = pd.cut(
        clean["prior_utilization"],
        bins=[-1, 0, 2, 5, np.inf],
        labels=["None", "Low", "Medium", "High"],
    )

    # The dataset's time_in_hospital values span 1-14 days, so these bins map
    # stays to 1-3, 4-7, and 8-14 days.
    clean["stay_group"] = pd.cut(
        clean["time_in_hospital"],
        bins=[0, 3, 7, 14],
        labels=["Short", "Medium", "Long"],
        include_lowest=True,
    )

    return clean


# -----------------------------------------------------------------------------
# 3. KPI summary
# -----------------------------------------------------------------------------
def calculate_kpis(df: pd.DataFrame) -> pd.Series:
    """Calculate the ten KPIs required by the capstone project."""
    kpis = {
        "Total Encounters": len(df),
        "Total Readmissions": int(df["readmission_flag"].sum()),
        "Readmission Rate (%)": df["readmission_flag"].mean() * 100,
        "Average Length of Stay": df["time_in_hospital"].mean(),
        "Average Medications": df["num_medications"].mean(),
        "Average Lab Procedures": df["num_lab_procedures"].mean(),
        "Average Prior Inpatient Visits": df["number_inpatient"].mean(),
        "Average Emergency Visits": df["number_emergency"].mean(),
        "Receiving Diabetes Medication (%)": df["diabetesMed"].eq("Yes").mean() * 100,
        "Medication Change (%)": df["change"].eq("Ch").mean() * 100,
    }
    return pd.Series(kpis, name="Value").round(2)


# -----------------------------------------------------------------------------
# 4. Advanced Pandas analysis
# -----------------------------------------------------------------------------
def run_pandas_analysis(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Create the notebook's patient, hospital, utilization, and medication summaries."""
    results: dict[str, pd.DataFrame] = {}

    # Q1-Q4: patient characteristics.
    results["gender"] = grouped_readmission_summary(df, "gender").sort_values(
        "Readmission_Rate", ascending=False
    )
    results["age"] = grouped_readmission_summary(df, "age").sort_values(
        "Readmission_Rate", ascending=False
    )
    results["race"] = grouped_readmission_summary(df, "race").sort_values(
        "Readmission_Rate", ascending=False
    )

    # Q5-Q7: hospitalization characteristics.
    results["admission"] = grouped_readmission_summary(df, "admission_type_id").sort_values(
        "Readmission_Rate", ascending=False
    )
    results["discharge"] = grouped_readmission_summary(
        df, "discharge_disposition_id"
    ).sort_values("Readmission_Rate", ascending=False)
    results["los"] = (
        df.groupby("readmission_flag", observed=True)
        .agg(Encounters=("rowID", "count"), Average_LOS=("time_in_hospital", "mean"))
        .reset_index()
    )
    results["los"]["Average_LOS"] = results["los"]["Average_LOS"].round(2)

    # Q8-Q11: prior healthcare utilization. The same helper keeps the
    # calculation consistent across inpatient, emergency, and outpatient visits.
    for name, column in {
        "inpatient": "number_inpatient",
        "emergency": "number_emergency",
        "outpatient": "number_outpatient",
        "overall_utilization": "utilization_group",
    }.items():
        results[name] = grouped_readmission_summary(df, column)

    # Q12: diagnosis ranking. Requiring at least 30 encounters avoids elevating
    # tiny diagnosis groups simply because they have an unstable high rate.
    diagnosis = grouped_readmission_summary(df, "diag_1_desc")
    diagnosis = diagnosis.loc[
        diagnosis["diag_1_desc"].notna()
        & diagnosis["Encounters"].ge(MIN_DIAGNOSIS_ENCOUNTERS)
    ].copy()
    diagnosis["Risk_Rank"] = diagnosis["Readmission_Rate"].rank(
        method="dense", ascending=False
    )
    results["diagnosis"] = diagnosis.sort_values(
        ["Readmission_Rate", "Encounters"], ascending=[False, False]
    )

    # Q13-Q14: diabetes medication status and medication change.
    results["diabetes_med"] = grouped_readmission_summary(df, "diabetesMed")
    results["medication_change"] = grouped_readmission_summary(df, "change")

    # Q15: multi-column segmentation. The volume threshold again prevents very
    # small age/admission combinations from dominating interpretation.
    age_admission = grouped_readmission_summary(df, ["age", "admission_type_id"])
    results["age_admission"] = (
        age_admission.loc[age_admission["Encounters"].ge(MIN_DIAGNOSIS_ENCOUNTERS)]
        .sort_values("Readmission_Rate", ascending=False)
        .reset_index(drop=True)
    )

    return results


def add_age_adjusted_los_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare each stay with the average LOS of the patient's age group.

    transform('mean') is appropriate because it returns a group-level average
    aligned to every original row, allowing a row-level difference to be built
    without merging a separate summary table back into the dataset.
    """
    df = df.copy()
    df["age_avg_los"] = df.groupby("age", observed=True)["time_in_hospital"].transform("mean")
    df["los_difference"] = df["time_in_hospital"] - df["age_avg_los"]
    df["los_vs_age_average"] = np.where(
        df["los_difference"].gt(0),
        "Above Age-Group Average",
        "At or Below Age-Group Average",
    )

    summary = grouped_readmission_summary(df, "los_vs_age_average")
    avg_los = (
        df.groupby("los_vs_age_average", observed=True)["time_in_hospital"]
        .mean()
        .rename("Average_LOS")
        .reset_index()
    )
    summary = summary.merge(avg_los, on="los_vs_age_average", how="left")
    return df, summary


def build_age_gender_pivot(df: pd.DataFrame) -> pd.DataFrame:
    """Return age-by-gender readmission percentages for analysis and heatmap use."""
    return pd.pivot_table(
        df,
        values="readmission_flag",
        index="age",
        columns="gender",
        aggfunc="mean",
        observed=True,
    ).mul(100)


# -----------------------------------------------------------------------------
# 5. Visualizations
# -----------------------------------------------------------------------------
def create_visualizations(
    df: pd.DataFrame,
    age_summary: pd.DataFrame,
    diagnosis_summary: pd.DataFrame,
    age_gender_pivot: pd.DataFrame,
    show: bool = True,
) -> None:
    """Recreate the seven visualizations from the notebook."""
    # 1) Readmission rate by age.
    age_plot = age_summary.sort_values("age")
    plt.figure(figsize=(10, 5))
    plt.bar(age_plot["age"].astype(str), age_plot["Readmission_Rate"])
    plt.title("Readmission Rate by Age Group")
    plt.xlabel("Age Group")
    plt.ylabel("Readmission Rate (%)")
    plt.xticks(rotation=45)
    plt.tight_layout()

    # 2) LOS by readmission status. A boxplot shows median, spread, and outliers,
    # which adds distributional context beyond comparing group means alone.
    plt.figure(figsize=(7, 5))
    df.boxplot(column="time_in_hospital", by="readmission_flag")
    plt.title("Length of Stay by Readmission Status")
    plt.suptitle("")
    plt.xlabel("Readmission Flag")
    plt.ylabel("Days")
    plt.tight_layout()

    # 3) Within-gender readmission distribution. Normalizing by row makes each
    # gender total 100%, so the chart compares proportions rather than counts.
    gender_readmission = pd.crosstab(
        df["gender"], df["readmission_flag"], normalize="index"
    ).mul(100)
    gender_readmission.plot(kind="bar", stacked=True, figsize=(7, 5))
    plt.title("Readmission Distribution by Gender")
    plt.xlabel("Gender")
    plt.ylabel("Percent of Encounters")
    plt.legend(["Not Readmitted", "Readmitted"])
    plt.tight_layout()

    # 4) Overall hospital LOS distribution.
    plt.figure(figsize=(8, 5))
    plt.hist(df["time_in_hospital"].dropna(), bins=14)
    plt.title("Distribution of Hospital Length of Stay")
    plt.xlabel("Days")
    plt.ylabel("Encounter Count")
    plt.tight_layout()

    # 5) Prior inpatient visits versus LOS.
    plt.figure(figsize=(8, 5))
    plt.scatter(df["number_inpatient"], df["time_in_hospital"], alpha=0.25)
    plt.title("Prior Inpatient Visits vs. Length of Stay")
    plt.xlabel("Prior Inpatient Visits")
    plt.ylabel("Length of Stay (Days)")
    plt.tight_layout()

    # 6) Age-gender heatmap using only Matplotlib.
    matrix = age_gender_pivot.to_numpy()
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(matrix, aspect="auto")
    ax.set_xticks(range(len(age_gender_pivot.columns)))
    ax.set_xticklabels(age_gender_pivot.columns)
    ax.set_yticks(range(len(age_gender_pivot.index)))
    ax.set_yticklabels(age_gender_pivot.index)
    ax.set_title("Readmission Rate by Age and Gender")
    ax.set_xlabel("Gender")
    ax.set_ylabel("Age Group")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            label = "NA" if np.isnan(value) else f"{value:.1f}%"
            ax.text(col, row, label, ha="center", va="center")
    fig.colorbar(image, ax=ax, label="Readmission Rate (%)")
    plt.tight_layout()

    # 7) Highest readmission-rate diagnoses after the 30-encounter threshold.
    top_diagnoses = (
        diagnosis_summary.sort_values("Readmission_Rate", ascending=False)
        .head(10)
        .sort_values("Readmission_Rate")
    )
    plt.figure(figsize=(10, 6))
    plt.barh(top_diagnoses["diag_1_desc"], top_diagnoses["Readmission_Rate"])
    plt.title(f"Top Primary Diagnoses by Readmission Rate ({MIN_DIAGNOSIS_ENCOUNTERS}+ Encounters)")
    plt.xlabel("Readmission Rate (%)")
    plt.ylabel("Primary Diagnosis")
    plt.tight_layout()

    if show:
        plt.show()
    else:
        plt.close("all")


# -----------------------------------------------------------------------------
# 6. SQLite analysis
# -----------------------------------------------------------------------------
def load_sqlite(df: pd.DataFrame, database_path: Path) -> sqlite3.Connection:
    """Load the cleaned encounter data into SQLite and return an open connection."""
    conn = sqlite3.connect(database_path)
    df.to_sql(SQL_TABLE, conn, if_exists="replace", index=False)
    return conn


def sql_queries() -> dict[str, str]:
    """Return the 14 SQL analyses from the notebook with descriptive names."""
    return {
        "1. Overall Readmission KPI": """
            SELECT
                COUNT(*) AS total_encounters,
                SUM(readmission_flag) AS total_readmissions,
                ROUND(100.0 * AVG(readmission_flag), 2) AS readmission_rate
            FROM hospital_encounters;
        """,
        "2. Readmission by Gender": """
            SELECT gender,
                   COUNT(*) AS encounters,
                   SUM(readmission_flag) AS readmissions,
                   ROUND(100.0 * AVG(readmission_flag), 2) AS readmission_rate
            FROM hospital_encounters
            GROUP BY gender
            ORDER BY readmission_rate DESC;
        """,
        "3. Readmission by Age": """
            SELECT age,
                   COUNT(*) AS encounters,
                   SUM(readmission_flag) AS readmissions,
                   ROUND(100.0 * AVG(readmission_flag), 2) AS readmission_rate
            FROM hospital_encounters
            GROUP BY age
            ORDER BY readmission_rate DESC;
        """,
        "4. Readmission by Race": """
            SELECT race,
                   COUNT(*) AS encounters,
                   SUM(readmission_flag) AS readmissions,
                   ROUND(100.0 * AVG(readmission_flag), 2) AS readmission_rate
            FROM hospital_encounters
            WHERE race IS NOT NULL
            GROUP BY race
            ORDER BY readmission_rate DESC;
        """,
        "5. Readmission by Admission Type": """
            SELECT admission_type_id,
                   COUNT(*) AS encounters,
                   SUM(readmission_flag) AS readmissions,
                   ROUND(100.0 * AVG(readmission_flag), 2) AS readmission_rate
            FROM hospital_encounters
            WHERE admission_type_id IS NOT NULL
            GROUP BY admission_type_id
            ORDER BY readmission_rate DESC;
        """,
        "6. LOS by Readmission Status": """
            SELECT readmission_flag,
                   COUNT(*) AS encounters,
                   ROUND(AVG(time_in_hospital), 2) AS average_los
            FROM hospital_encounters
            GROUP BY readmission_flag;
        """,
        "7. Prior Inpatient Utilization": """
            SELECT number_inpatient,
                   COUNT(*) AS encounters,
                   SUM(readmission_flag) AS readmissions,
                   ROUND(100.0 * AVG(readmission_flag), 2) AS readmission_rate
            FROM hospital_encounters
            GROUP BY number_inpatient
            ORDER BY number_inpatient;
        """,
        "8. Utilization CASE WHEN": """
            SELECT
                CASE
                    WHEN prior_utilization = 0 THEN 'None'
                    WHEN prior_utilization BETWEEN 1 AND 2 THEN 'Low'
                    WHEN prior_utilization BETWEEN 3 AND 5 THEN 'Medium'
                    ELSE 'High'
                END AS utilization_group_sql,
                COUNT(*) AS encounters,
                SUM(readmission_flag) AS readmissions,
                ROUND(100.0 * AVG(readmission_flag), 2) AS readmission_rate
            FROM hospital_encounters
            GROUP BY utilization_group_sql
            ORDER BY readmission_rate DESC;
        """,
        "9. Diagnosis with HAVING": f"""
            SELECT diag_1_desc,
                   COUNT(*) AS encounters,
                   SUM(readmission_flag) AS readmissions,
                   ROUND(100.0 * AVG(readmission_flag), 2) AS readmission_rate
            FROM hospital_encounters
            WHERE diag_1_desc IS NOT NULL
            GROUP BY diag_1_desc
            HAVING COUNT(*) >= {MIN_DIAGNOSIS_ENCOUNTERS}
            ORDER BY readmission_rate DESC;
        """,
        "10. Diagnosis CTE": f"""
            WITH diagnosis_summary AS (
                SELECT diag_1_desc,
                       COUNT(*) AS encounters,
                       SUM(readmission_flag) AS readmissions,
                       100.0 * AVG(readmission_flag) AS readmission_rate
                FROM hospital_encounters
                WHERE diag_1_desc IS NOT NULL
                GROUP BY diag_1_desc
            )
            SELECT diag_1_desc,
                   encounters,
                   readmissions,
                   ROUND(readmission_rate, 2) AS readmission_rate
            FROM diagnosis_summary
            WHERE encounters >= {MIN_DIAGNOSIS_ENCOUNTERS}
            ORDER BY readmission_rate DESC;
        """,
        "11. Diagnosis Window Function and Ranking": f"""
            WITH diagnosis_summary AS (
                SELECT diag_1_desc,
                       COUNT(*) AS encounters,
                       100.0 * AVG(readmission_flag) AS readmission_rate
                FROM hospital_encounters
                WHERE diag_1_desc IS NOT NULL
                GROUP BY diag_1_desc
                HAVING COUNT(*) >= {MIN_DIAGNOSIS_ENCOUNTERS}
            )
            SELECT diag_1_desc,
                   encounters,
                   ROUND(readmission_rate, 2) AS readmission_rate,
                   DENSE_RANK() OVER (ORDER BY readmission_rate DESC) AS risk_rank
            FROM diagnosis_summary
            ORDER BY risk_rank, encounters DESC;
        """,
        "12. Conditional Aggregation": """
            SELECT
                COUNT(*) AS total_encounters,
                SUM(CASE WHEN readmission_flag = 1 THEN 1 ELSE 0 END) AS readmissions,
                SUM(CASE WHEN readmission_flag = 0 THEN 1 ELSE 0 END) AS not_readmitted,
                ROUND(
                    100.0 * SUM(CASE WHEN readmission_flag = 1 THEN 1 ELSE 0 END) / COUNT(*),
                    2
                ) AS readmission_rate
            FROM hospital_encounters;
        """,
        "13. Diabetes Medication": """
            SELECT diabetesMed,
                   COUNT(*) AS encounters,
                   SUM(readmission_flag) AS readmissions,
                   ROUND(100.0 * AVG(readmission_flag), 2) AS readmission_rate
            FROM hospital_encounters
            GROUP BY diabetesMed
            ORDER BY readmission_rate DESC;
        """,
        "14. Medication Change": """
            SELECT change,
                   COUNT(*) AS encounters,
                   SUM(readmission_flag) AS readmissions,
                   ROUND(100.0 * AVG(readmission_flag), 2) AS readmission_rate
            FROM hospital_encounters
            GROUP BY change
            ORDER BY readmission_rate DESC;
        """,
    }


def run_sql_analysis(conn: sqlite3.Connection) -> dict[str, pd.DataFrame]:
    """Execute all SQL analyses and return their result tables."""
    return {name: pd.read_sql_query(query, conn) for name, query in sql_queries().items()}


# -----------------------------------------------------------------------------
# 7. Pandas vs. SQL validation
# -----------------------------------------------------------------------------
def validate_pandas_vs_sql(df: pd.DataFrame, conn: sqlite3.Connection) -> pd.DataFrame:
    """Independently calculate five KPIs in Pandas and SQL and compare them."""
    pandas_values = {
        "Total Encounters": len(df),
        "Total Readmissions": df["readmission_flag"].sum(),
        "Readmission Rate": df["readmission_flag"].mean() * 100,
        "Average Length of Stay": df["time_in_hospital"].mean(),
        "Average Medications": df["num_medications"].mean(),
    }

    sql_values = pd.read_sql_query(
        """
        SELECT
            COUNT(*) AS total_encounters,
            SUM(readmission_flag) AS total_readmissions,
            100.0 * AVG(readmission_flag) AS readmission_rate,
            AVG(time_in_hospital) AS average_los,
            AVG(num_medications) AS average_medications
        FROM hospital_encounters;
        """,
        conn,
    ).iloc[0]

    sql_map = {
        "Total Encounters": sql_values["total_encounters"],
        "Total Readmissions": sql_values["total_readmissions"],
        "Readmission Rate": sql_values["readmission_rate"],
        "Average Length of Stay": sql_values["average_los"],
        "Average Medications": sql_values["average_medications"],
    }

    validation = pd.DataFrame(
        {
            "KPI": list(pandas_values),
            "Pandas Result": list(pandas_values.values()),
            "SQL Result": [sql_map[k] for k in pandas_values],
        }
    )
    validation["Match"] = np.isclose(
        validation["Pandas Result"], validation["SQL Result"], equal_nan=True
    )
    return validation


# -----------------------------------------------------------------------------
# Main workflow
# -----------------------------------------------------------------------------
def main(input_path: Path, clean_csv: Path, database_path: Path, show_plots: bool) -> None:
    """Run the complete CRISP-DM-style descriptive analytics workflow."""
    section("1. DATA UNDERSTANDING")
    raw = load_data(input_path)
    print(f"Shape: {raw.shape}")
    print(f"Duplicate rows: {raw.duplicated().sum():,}")
    print(f"Unique row IDs: {raw['rowID'].nunique():,}")
    print(f"Total rows: {len(raw):,}")

    profile, question_marks = data_quality_profile(raw)
    show_table("Top 20 columns by native missingness", profile, rows=20)
    show_table("Columns containing '?' missing-value markers", question_marks)

    section("2. DATA PREPARATION AND FEATURE ENGINEERING")
    clean = prepare_data(raw)
    missing_after = clean.isna().mean().mul(100).sort_values(ascending=False)
    show_table("Top 15 columns by missingness after cleaning", missing_after, rows=15)

    section("3. KPI SUMMARY")
    kpis = calculate_kpis(clean)
    show_table("Required KPIs", kpis)

    section("4. ADVANCED PANDAS ANALYSIS")
    pandas_results = run_pandas_analysis(clean)
    clean, los_age_summary = add_age_adjusted_los_features(clean)
    age_gender_pivot = build_age_gender_pivot(clean)

    # Print the most decision-relevant summaries. The remaining tables remain
    # available in pandas_results for programmatic use.
    show_table("Readmission by gender", pandas_results["gender"])
    show_table("Readmission by age", pandas_results["age"])
    show_table("Readmission by race", pandas_results["race"])
    show_table("Readmission by admission type", pandas_results["admission"])
    show_table("LOS by readmission status", pandas_results["los"])
    show_table("Prior inpatient visits", pandas_results["inpatient"], rows=10)
    show_table("Prior emergency visits", pandas_results["emergency"], rows=10)
    show_table("Prior outpatient visits", pandas_results["outpatient"], rows=10)
    show_table("Overall utilization groups", pandas_results["overall_utilization"])
    show_table("Top diagnosis groups (30+ encounters)", pandas_results["diagnosis"], rows=15)
    show_table("Diabetes medication status", pandas_results["diabetes_med"])
    show_table("Medication change", pandas_results["medication_change"])
    show_table("Age + admission-type segments", pandas_results["age_admission"], rows=15)
    show_table("LOS compared with age-group average", los_age_summary)
    show_table("Age-gender readmission pivot (%)", age_gender_pivot.round(2))

    # .loc example retained from the notebook: select readmitted encounters with
    # stays of at least seven days and only the columns needed for inspection.
    long_readmitted = clean.loc[
        clean["readmission_flag"].eq(1) & clean["time_in_hospital"].ge(7),
        [
            "age",
            "gender",
            "time_in_hospital",
            "number_inpatient",
            "number_emergency",
            "diag_1_desc",
        ],
    ]
    show_table("Example: readmitted encounters with LOS >= 7 days", long_readmitted, rows=5)

    section("5. VISUALIZATIONS")
    create_visualizations(
        clean,
        pandas_results["age"],
        pandas_results["diagnosis"],
        age_gender_pivot,
        show=show_plots,
    )

    section("6. EXPORT CLEANED DATA")
    clean.to_csv(clean_csv, index=False)
    print(f"Saved cleaned CSV: {clean_csv}")

    section("7. SQL ANALYSIS")
    conn = load_sqlite(clean, database_path)
    print(f"Loaded cleaned data into SQLite table '{SQL_TABLE}' in {database_path}")

    try:
        for name, result in run_sql_analysis(conn).items():
            show_table(name, result, rows=15)

        section("8. PANDAS VS. SQL VALIDATION")
        validation = validate_pandas_vs_sql(clean, conn)
        show_table("KPI validation", validation.round(2))
        print(f"\nAll five KPIs match: {bool(validation['Match'].all())}")
    finally:
        # Explicitly close the database connection even if a query fails.
        conn.close()

    section("ANALYSIS COMPLETE")
    print(
        "Interpret readmission differences as descriptive associations, not causal effects. "
        "Prioritize segments that combine elevated readmission rates with meaningful encounter volume."
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line paths while keeping notebook-compatible defaults."""
    parser = argparse.ArgumentParser(description="Diabetes hospital readmission analytics")
    parser.add_argument("--input", type=Path, default=Path(DEFAULT_INPUT), help="Raw CSV path")
    parser.add_argument(
        "--clean-output", type=Path, default=Path(DEFAULT_CLEAN_CSV), help="Cleaned CSV path"
    )
    parser.add_argument(
        "--database", type=Path, default=Path(DEFAULT_DATABASE), help="SQLite database path"
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Run the analysis without opening Matplotlib plot windows",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        input_path=args.input,
        clean_csv=args.clean_output,
        database_path=args.database,
        show_plots=not args.no_plots,
    )
