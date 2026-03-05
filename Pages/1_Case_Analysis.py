import pandas as pd
import plotly.express as px
import streamlit as st

from helpers import AGE_LABELS, DISEASE_COLUMNS, apply_light_theme, filter_dataset, load_preprocessed_data, render_logo

st.set_page_config(page_title="Case Analysis", layout="wide")
apply_light_theme()

render_logo(220)
st.title("Case Analysis")

try:
    df = load_preprocessed_data()

    st.sidebar.header("Filters")
    outcomes = ["ALL", "POSITIVE", "NEGATIVE", "PENDING"]
    if "OUTCOME" in df.columns:
        observed_outcomes = sorted(df["OUTCOME"].dropna().astype(str).str.upper().unique().tolist())
        outcomes.extend([o for o in observed_outcomes if o not in outcomes])

    selected_outcome = st.sidebar.selectbox("Outcome", outcomes)

    min_age = int(df["AGE_NUM"].min(skipna=True))
    max_age = int(df["AGE_NUM"].max(skipna=True))
    selected_age = st.sidebar.slider("Age Range", min_age, max_age, (min_age, max_age))

    disease_options = [d for d in DISEASE_COLUMNS if d in df.columns]

    filtered = filter_dataset(df, selected_outcome, selected_age)

    st.write(f"Filtered records: {len(filtered):,}")

    t1, t2, t3, t4, t5 = st.tabs([
        "Age Susceptibility",
        "Gender and Age Distribution",
        "Patients Required Intubation",
        "Disease and ICU Correlation",
        "Deceased Comorbidities",
    ])

    with t1:
        st.subheader("Age group susceptibility to COVID-19")
        age_counts = (
            filtered["AGE_GROUP"]
            .value_counts(sort=False)
            .reindex(AGE_LABELS, fill_value=0)
            .rename_axis("AGE_GROUP")
            .reset_index(name="COUNT")
        )
        fig1 = px.bar(age_counts, x="AGE_GROUP", y="COUNT", title="Cases by Age Group")
        st.plotly_chart(fig1, use_container_width=True)

    with t2:
        st.subheader("Distribution of cases by gender and age group")
        if "SEX" in filtered.columns:
            pivot = pd.crosstab(filtered["AGE_GROUP"], filtered["SEX"]).reindex(AGE_LABELS, fill_value=0)
            if not pivot.empty:
                fig2 = px.imshow(
                    pivot,
                    text_auto=True,
                    aspect="auto",
                    title="Heatmap: Age Group vs Gender",
                )
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("No data after filters.")
        else:
            st.info("SEX column not available.")

    with t3:
        st.subheader("Patients required intubation")
        if "INTUBATED" in filtered.columns:
            intub_counts = filtered["INTUBATED"].astype(str).str.upper().value_counts().reset_index()
            intub_counts.columns = ["INTUBATED", "COUNT"]
            yes_count = int((filtered["INTUBATED"].astype(str).str.upper() == "YES").sum())
            st.metric("Patients requiring intubation", f"{yes_count:,}")
            intub_order = ["YES", "NO", "DOES NOT APPLY", "UNKNOWN", "IGNORED"]
            intub_counts["ORDER"] = intub_counts["INTUBATED"].apply(
                lambda x: intub_order.index(x) if x in intub_order else len(intub_order)
            )
            intub_counts = intub_counts.sort_values("ORDER").drop(columns=["ORDER"])
            fig3 = px.area(
                intub_counts,
                x="INTUBATED",
                y="COUNT",
                markers=True,
                title="Intubation Distribution (Area Chart)",
            ).update_layout(height=420)
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("INTUBATED column not available.")

    with t4:
        st.subheader("Correlation Between Diseases and ICU Admission")
        if "ICU" in filtered.columns and disease_options:
            icu_yes_mask = filtered["ICU"].astype(str).str.upper().eq("YES")
            q4_rows = []
            for d in disease_options:
                disease_yes_mask = filtered[d].astype(str).str.upper().eq("YES")
                disease_no_mask = ~disease_yes_mask

                total_with_disease = int(disease_yes_mask.sum())
                total_without_disease = int(disease_no_mask.sum())
                icu_yes_with_disease = int((disease_yes_mask & icu_yes_mask).sum())
                icu_yes_without_disease = int((disease_no_mask & icu_yes_mask).sum())

                icu_rate_with = (icu_yes_with_disease / total_with_disease) if total_with_disease > 0 else 0.0
                icu_rate_without = (
                    icu_yes_without_disease / total_without_disease if total_without_disease > 0 else 0.0
                )
                q4_rows.append(
                    {
                        "DISEASE": d,
                        "TOTAL_WITH_DISEASE": total_with_disease,
                        "ICU_YES_WITH_DISEASE": icu_yes_with_disease,
                        "ICU_RATE_WITH_DISEASE": icu_rate_with,
                        "TOTAL_WITHOUT_DISEASE": total_without_disease,
                        "ICU_YES_WITHOUT_DISEASE": icu_yes_without_disease,
                        "ICU_RATE_WITHOUT_DISEASE": icu_rate_without,
                        "RATE_DIFFERENCE": icu_rate_with - icu_rate_without,
                    }
                )

            q4_table = pd.DataFrame(q4_rows).sort_values("RATE_DIFFERENCE", ascending=False)

            q4_plot = q4_table.melt(
                id_vars=["DISEASE"],
                value_vars=["ICU_RATE_WITH_DISEASE", "ICU_RATE_WITHOUT_DISEASE"],
                var_name="GROUP",
                value_name="ICU_RATE",
            )
            q4_plot["GROUP"] = q4_plot["GROUP"].replace(
                {
                    "ICU_RATE_WITH_DISEASE": "With disease",
                    "ICU_RATE_WITHOUT_DISEASE": "Without disease",
                }
            )

            fig4 = px.bar(
                q4_plot.assign(ICU_RATE_PCT=q4_plot["ICU_RATE"] * 100),
                x="DISEASE",
                y="ICU_RATE_PCT",
                color="GROUP",
                barmode="group",
                title="ICU Admission Rate by Disease (With vs Without Disease)",
                labels={"DISEASE": "Disease", "ICU_RATE_PCT": "ICU Rate (%)", "GROUP": "Group"},
            )
            st.plotly_chart(fig4, use_container_width=True)
            st.dataframe(q4_table, use_container_width=True)
        else:
            st.info("ICU or disease columns are missing, or no diseases selected.")

    with t5:
        st.subheader("Common diseases among deceased patients")
        if "DATE_OF_DEATH" in filtered.columns and disease_options:
            deceased = filtered[filtered["DATE_OF_DEATH"].notna()]
            st.metric("Deceased records", f"{len(deceased):,}")

            if len(deceased) > 0:
                rows = []
                for d in disease_options:
                    rows.append({"DISEASE": d, "COUNT": int((deceased[d].astype(str).str.upper() == "YES").sum())})
                disease_df = pd.DataFrame(rows).sort_values("COUNT", ascending=False)

                fig5 = px.bar(
                    disease_df,
                    x="COUNT",
                    y="DISEASE",
                    orientation="h",
                    title="Common Diseases in Deceased Patients",
                )
                st.plotly_chart(fig5, use_container_width=True)
                st.dataframe(disease_df, use_container_width=True)
            else:
                st.info("No deceased records in current filtered dataset.")
        else:
            st.info("DATE_OF_DEATH or disease columns not available.")

except Exception as exc:
    st.error(f"Analysis page failed: {exc}")
