import re
from pathlib import Path

import pandas as pd
import streamlit as st

DATASET_PATH = Path("dataset.csv")
DICTIONARY_CSV_PATH = Path("data_dictionary.csv")
DICTIONARY_XLSX_PATH = Path("data_dictionary.xlsx")
LOGO_PATH = Path("fourtitude_logo.png")

AGE_BINS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
AGE_LABELS = [
    "0-10",
    "10-20",
    "20-30",
    "30-40",
    "40-50",
    "50-60",
    "60-70",
    "70-80",
    "80-90",
    "90-100",
]

DISEASE_COLUMNS = [
    "DIABETES",
    "COPD",
    "ASTHMA",
    "INMUSUPR",
    "HYPERTENSION",
    "CARDIOVASCULAR",
    "OBESITY",
    "CHRONIC_KIDNEY",
    "TOBACCO",
]


@st.cache_data
def normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).strip().lower())


@st.cache_data
def parse_mapping_text(text: str) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for part in str(text).split(","):
        part = part.strip()
        if "=" not in part:
            continue
        key_raw, label_raw = part.split("=", 1)
        key_match = re.search(r"\d+", key_raw.strip())
        if not key_match:
            continue
        mapping[int(key_match.group())] = label_raw.strip().upper()
    return mapping


@st.cache_data
def load_dictionary_mappings() -> dict[str, dict[int, str]]:
    if DICTIONARY_CSV_PATH.exists():
        dd = pd.read_csv(DICTIONARY_CSV_PATH)
    elif DICTIONARY_XLSX_PATH.exists():
        dd = pd.read_excel(DICTIONARY_XLSX_PATH)
    else:
        raise FileNotFoundError("Expected data_dictionary.csv or data_dictionary.xlsx")

    dd = dd.rename(columns={c: str(c).strip().lower() for c in dd.columns})
    if "variable" not in dd.columns or "value" not in dd.columns:
        raise ValueError("Data dictionary must contain 'variable' and 'value' columns")

    mappings: dict[str, dict[int, str]] = {}
    for _, row in dd.iterrows():
        if pd.isna(row["variable"]) or pd.isna(row["value"]):
            continue
        parsed = parse_mapping_text(row["value"])
        if parsed:
            mappings[normalize_name(row["variable"])] = parsed

    return mappings


@st.cache_data
def load_preprocessed_data() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise FileNotFoundError("dataset.csv was not found")

    df = pd.read_csv(DATASET_PATH)
    mappings = load_dictionary_mappings()
    out = df.copy()
    normalized_col_lookup = {normalize_name(col): col for col in out.columns}

    for normalized_var, value_map in mappings.items():
        if normalized_var not in normalized_col_lookup:
            continue
        col = normalized_col_lookup[normalized_var]
        numeric_series = pd.to_numeric(out[col], errors="coerce")
        out[col] = numeric_series.map(value_map).fillna(out[col])

    out["AGE_NUM"] = pd.to_numeric(out["AGE"], errors="coerce")
    out["AGE_GROUP"] = pd.cut(
        out["AGE_NUM"],
        bins=AGE_BINS,
        labels=AGE_LABELS,
        include_lowest=True,
        right=True,
    )
    return out


def yes_as_int(series: pd.Series) -> pd.Series:
    return (series.astype(str).str.upper() == "YES").astype(int)


def render_logo(width: int = 260) -> None:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=width)


def apply_light_theme() -> None:
    st.markdown(
        """
        <style>
            :root { color-scheme: light; }
            .stApp {
                background-color: #ffffff;
                color: #111111;
            }
            [data-testid="stSidebar"] {
                background-color: #f6f8fc;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def filter_dataset(df: pd.DataFrame, outcome_filter: str, age_range: tuple[int, int]) -> pd.DataFrame:
    out = df.copy()
    out = out[out["AGE_NUM"].between(age_range[0], age_range[1], inclusive="both")]
    if outcome_filter != "ALL" and "OUTCOME" in out.columns:
        out = out[out["OUTCOME"].astype(str).str.upper() == outcome_filter]
    return out


def build_text_report(df: pd.DataFrame) -> str:
    age_counts = df["AGE_GROUP"].value_counts(sort=False).reindex(AGE_LABELS, fill_value=0)
    intubated_yes = int((df["INTUBATED"].astype(str).str.upper() == "YES").sum()) if "INTUBATED" in df.columns else 0
    total = len(df)

    corr_lines = []
    if "ICU" in df.columns:
        temp = pd.DataFrame({"ICU": yes_as_int(df["ICU"])})
        for col in DISEASE_COLUMNS:
            if col in df.columns:
                temp[col] = yes_as_int(df[col])
        corr = temp.corr(numeric_only=True).get("ICU", pd.Series(dtype=float)).drop("ICU", errors="ignore")
        corr_lines = [f"- {k}: {v:.3f}" for k, v in corr.sort_values(ascending=False).items()]

    deceased_lines = []
    if "DATE_OF_DEATH" in df.columns:
        deceased = df[df["DATE_OF_DEATH"].notna()]
        if len(deceased):
            disease_counts = {
                d: int((deceased[d].astype(str).str.upper() == "YES").sum())
                for d in DISEASE_COLUMNS
                if d in deceased.columns
            }
            ranked = sorted(disease_counts.items(), key=lambda x: x[1], reverse=True)
            deceased_lines = [f"- {d}: {c}" for d, c in ranked]

    lines = [
        "COVID-19 Dashboard Report",
        "",
        f"Total filtered records: {total}",
        "",
        "Q1: Susceptible age groups (counts)",
        *[f"- {k}: {int(v)}" for k, v in age_counts.items() if pd.notna(k)],
        "",
        f"Q3: Patients requiring intubation: {intubated_yes}",
        "",
        "Q4: Correlation with ICU admission",
        *corr_lines,
        "",
        "Q5: Common diseases among deceased patients",
        *deceased_lines,
    ]
    return "\n".join(lines)
