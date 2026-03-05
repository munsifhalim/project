import streamlit as st

from helpers import apply_light_theme, load_preprocessed_data, render_logo

st.set_page_config(page_title="COVID-19 Dashboard", layout="wide")
apply_light_theme()

render_logo()
st.title("COVID-19 Interactive Dashboard")
st.caption("Fourtitude training dashboard with preprocessing from the data dictionary.")

try:
    df = load_preprocessed_data()

    outcome_series = df["OUTCOME"].astype(str).str.upper() if "OUTCOME" in df.columns else None
    positive_count = int((outcome_series == "POSITIVE").sum()) if outcome_series is not None else 0
    negative_count = int((outcome_series == "NEGATIVE").sum()) if outcome_series is not None else 0
    pending_count = int((outcome_series == "PENDING").sum()) if outcome_series is not None else 0
    all_three_count = positive_count + negative_count + pending_count

    st.subheader("COVID-19 Case Counts")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Cases", f"{all_three_count:,}")
    m2.metric("Positive", f"{positive_count:,}")
    m3.metric("Negative", f"{negative_count:,}")
    m4.metric("Pending", f"{pending_count:,}")

    tab1, tab2 = st.tabs(["Overview", "Data Preview"])

    with tab1:
        st.write("Use the left sidebar pages:")
        st.write("1. Case Analysis:")
        st.write("- Age group susceptibility to COVID-19")
        st.write("- Distribution of cases by gender and age group")
        st.write("- Patients required intubation")
        st.write("- Correlation between diseases and ICU admission")
        st.write("- Common diseases among deceased patients")
        st.write("2. Consolidated Report")

    with tab2:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Records", f"{len(df):,}")
        c2.metric("Columns", f"{df.shape[1]:,}")
        c3.metric("Age Range", f"{int(df['AGE_NUM'].min(skipna=True))} - {int(df['AGE_NUM'].max(skipna=True))}")
        st.dataframe(df.head(50), use_container_width=True)

except Exception as exc:
    st.error(f"Failed to load dashboard data: {exc}")
