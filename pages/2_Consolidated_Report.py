from datetime import datetime
from io import BytesIO

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Image as RLImage
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

from helpers import (
    AGE_LABELS,
    DISEASE_COLUMNS,
    LOGO_PATH,
    apply_light_theme,
    filter_dataset,
    load_preprocessed_data,
    render_logo,
)

st.set_page_config(page_title="Consolidated Report", layout="wide")
apply_light_theme()


def table_from_df(df: pd.DataFrame, max_rows: int = 20, fit_width: float | None = None, font_size: int = 8) -> Table:
    safe_df = df.head(max_rows).copy()
    data = [safe_df.columns.tolist()] + safe_df.astype(str).values.tolist()
    col_widths = None
    if fit_width:
        col_widths = [fit_width / max(len(safe_df.columns), 1)] * len(safe_df.columns)
    table = Table(data, repeatRows=1, colWidths=col_widths)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e9f2ff")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), font_size),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    table.hAlign = "LEFT"
    return table


def table_from_df_custom_widths(df: pd.DataFrame, col_widths: list[float], max_rows: int = 20, font_size: int = 8) -> Table:
    safe_df = df.head(max_rows).copy()
    data = [safe_df.columns.tolist()] + safe_df.astype(str).values.tolist()
    table = Table(data, repeatRows=1, colWidths=col_widths)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e9f2ff")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), font_size),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    table.hAlign = "LEFT"
    return table


def fig_to_bytes(fig) -> BytesIO:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def chart_age_counts(age_counts: pd.DataFrame) -> BytesIO:
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.bar(age_counts["AGE_GROUP"], age_counts["COUNT"], color="#4a90e2")
    ax.set_title("Cases by Age Group")
    ax.set_xlabel("Age Group")
    ax.set_ylabel("Count")
    plt.xticks(rotation=45)
    return fig_to_bytes(fig)


def chart_gender_heatmap(gender_age: pd.DataFrame) -> BytesIO:
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    img = ax.imshow(gender_age.values, aspect="auto", cmap="Blues")
    ax.set_title("Age Group vs Gender")
    ax.set_yticks(range(len(gender_age.index)))
    ax.set_yticklabels([str(x) for x in gender_age.index.tolist()])
    ax.set_xticks(range(len(gender_age.columns)))
    ax.set_xticklabels([str(x) for x in gender_age.columns.tolist()], rotation=30, ha="right")
    fig.colorbar(img, ax=ax, fraction=0.045, pad=0.04)
    return fig_to_bytes(fig)


def chart_intubation_area(intub_counts: pd.DataFrame) -> BytesIO:
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    x = list(range(len(intub_counts)))
    y = intub_counts["COUNT"].tolist()
    ax.plot(x, y, color="#4a90e2", linewidth=2, marker="o")
    ax.fill_between(x, y, color="#4a90e2", alpha=0.25)
    ax.set_xticks(x)
    ax.set_xticklabels(intub_counts["INTUBATED"].tolist(), rotation=20, ha="right")
    ax.set_title("Intubation Distribution (Area Chart)")
    ax.set_xlabel("Intubation")
    ax.set_ylabel("Count")
    return fig_to_bytes(fig)


def chart_icu_correlation(q4_table: pd.DataFrame) -> BytesIO:
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    q4_sorted = q4_table.sort_values("RATE_DIFFERENCE", ascending=False)
    x = range(len(q4_sorted))
    width = 0.38
    ax.bar(
        [i - width / 2 for i in x],
        q4_sorted["ICU_RATE_WITH_DISEASE"] * 100,
        width=width,
        color="#35a76d",
        label="With disease",
    )
    ax.bar(
        [i + width / 2 for i in x],
        q4_sorted["ICU_RATE_WITHOUT_DISEASE"] * 100,
        width=width,
        color="#8aaed1",
        label="Without disease",
    )
    ax.set_title("ICU Admission Rate by Disease (With vs Without Disease)")
    ax.set_xlabel("Disease")
    ax.set_ylabel("ICU Rate (%)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(q4_sorted["DISEASE"])
    ax.legend(frameon=False, fontsize=8)
    plt.xticks(rotation=35, ha="right")
    return fig_to_bytes(fig)


def chart_deceased(deceased_df: pd.DataFrame) -> BytesIO:
    fig, ax = plt.subplots(figsize=(6.0, 3.7))
    ax.barh(deceased_df["DISEASE"], deceased_df["COUNT"], color="#e07a5f")
    ax.set_title("Common Diseases in Deceased Patients")
    ax.set_xlabel("Count")
    return fig_to_bytes(fig)


def build_pdf_report(
    exec_summary: list[str],
    age_counts: pd.DataFrame,
    gender_age: pd.DataFrame,
    intub_counts: pd.DataFrame,
    intub_yes: int,
    q4_table: pd.DataFrame,
    deceased_df: pd.DataFrame,
) -> bytes:
    buffer = BytesIO()
    doc = BaseDocTemplate(buffer, pagesize=A4, leftMargin=72, rightMargin=72, topMargin=72, bottomMargin=72)
    styles = getSampleStyleSheet()
    elements = []
    content_width = A4[0] - doc.leftMargin - doc.rightMargin

    logo_reader = ImageReader(str(LOGO_PATH)) if LOGO_PATH.exists() else None
    logo_w, logo_h = logo_reader.getSize() if logo_reader else (1, 1)

    def left_image(image_source, width: float, height: float) -> RLImage:
        img = RLImage(image_source, width=width, height=height)
        img.hAlign = "LEFT"
        return img

    def draw_cover(canvas, _doc):
        canvas.saveState()
        page_w, page_h = A4
        title = "COVID-19 Consolidated Report"
        date_text = datetime.now().strftime("Generated on %Y-%m-%d %H:%M:%S")

        cover_w = 250 if logo_reader else 0
        cover_h = cover_w * (logo_h / logo_w) if logo_reader else 0
        block_h = cover_h + 24 + 20 + 20
        block_bottom = (page_h - block_h) / 2

        if logo_reader:
            canvas.drawImage(
                logo_reader,
                (page_w - cover_w) / 2,
                block_bottom + 44,
                width=cover_w,
                height=cover_h,
                preserveAspectRatio=True,
                mask="auto",
            )

        canvas.setFont("Helvetica-Bold", 22)
        canvas.setFillColor(colors.black)
        canvas.drawCentredString(page_w / 2, block_bottom + 20, title)
        canvas.setFont("Helvetica", 12)
        canvas.setFillColor(colors.HexColor("#333333"))
        canvas.drawCentredString(page_w / 2, block_bottom, date_text)
        canvas.restoreState()

    def draw_footer(canvas, _doc):
        canvas.saveState()
        y = 14
        x = _doc.leftMargin
        if logo_reader:
            footer_h = 10
            footer_w = footer_h * (logo_w / logo_h)
            if hasattr(canvas, "setFillAlpha"):
                canvas.setFillAlpha(0.45)
            canvas.drawImage(
                logo_reader,
                x,
                y - 1,
                width=footer_w,
                height=footer_h,
                preserveAspectRatio=True,
                mask="auto",
            )
        canvas.restoreState()

    portrait_frame = Frame(doc.leftMargin, doc.bottomMargin, content_width, A4[1] - doc.topMargin - doc.bottomMargin, id="portrait_frame")
    doc.addPageTemplates(
        [
            PageTemplate(id="cover", frames=[portrait_frame], onPage=draw_cover, pagesize=A4),
            PageTemplate(id="portrait", frames=[portrait_frame], onPage=draw_footer, pagesize=A4),
        ]
    )

    # Page 1: Cover (drawn by template callback), start flowables from page 2.
    elements.append(NextPageTemplate("portrait"))
    elements.append(PageBreak())

    # Page 2: Table of Contents
    elements.append(Paragraph("Table of Contents", styles["Heading2"]))
    elements.append(Spacer(1, 8))
    elements.append(Paragraph("1. Age group susceptibility to COVID-19", styles["Normal"]))
    elements.append(Paragraph("2. Distribution of cases by gender and age group", styles["Normal"]))
    elements.append(Paragraph("3. Patients required intubation", styles["Normal"]))
    elements.append(Paragraph("4. Correlation between diseases and ICU admission", styles["Normal"]))
    elements.append(Paragraph("5. Common diseases among deceased patients", styles["Normal"]))
    elements.append(Spacer(1, 14))
    elements.append(Paragraph("Executive Summary", styles["Heading3"]))
    for line in exec_summary:
        elements.append(Paragraph(f"- {line}", styles["Normal"]))
    elements.append(PageBreak())

    # Page 3+: One page per case
    elements.append(Paragraph("Age group susceptibility to COVID-19", styles["Heading3"]))
    elements.append(left_image(chart_age_counts(age_counts), content_width, 255))
    elements.append(Spacer(1, 6))
    elements.append(table_from_df(age_counts))
    elements.append(PageBreak())

    elements.append(Paragraph("Distribution of cases by gender and age group", styles["Heading3"]))
    elements.append(left_image(chart_gender_heatmap(gender_age), content_width, 265))
    elements.append(Spacer(1, 6))
    elements.append(table_from_df(gender_age.reset_index()))
    elements.append(PageBreak())

    elements.append(Paragraph("Patients required intubation", styles["Heading3"]))
    elements.append(Paragraph(f"Patients Required Intubation: {intub_yes:,}", styles["Normal"]))
    elements.append(left_image(chart_intubation_area(intub_counts), 400, 213))
    elements.append(Spacer(1, 6))
    elements.append(table_from_df(intub_counts))
    elements.append(PageBreak())

    elements.append(Paragraph("Correlation between diseases and ICU admission", styles["Heading3"]))
    elements.append(left_image(chart_icu_correlation(q4_table), content_width, 300))
    elements.append(Spacer(1, 6))
    q4_pdf = q4_table[
        [
            "DISEASE",
            "TOTAL_WITH_DISEASE",
            "ICU_YES_WITH_DISEASE",
            "ICU_RATE_WITH_DISEASE",
            "TOTAL_WITHOUT_DISEASE",
            "ICU_YES_WITHOUT_DISEASE",
            "ICU_RATE_WITHOUT_DISEASE",
            "RATE_DIFFERENCE",
        ]
    ].copy()
    q4_pdf["ICU_RATE_WITH_DISEASE"] = (q4_pdf["ICU_RATE_WITH_DISEASE"] * 100).round(2)
    q4_pdf["ICU_RATE_WITHOUT_DISEASE"] = (q4_pdf["ICU_RATE_WITHOUT_DISEASE"] * 100).round(2)
    q4_pdf["RATE_DIFFERENCE"] = (q4_pdf["RATE_DIFFERENCE"] * 100).round(2)
    q4_pdf = q4_pdf.rename(
        columns={
            "DISEASE": "Disease",
            "TOTAL_WITH_DISEASE": "N_With",
            "ICU_YES_WITH_DISEASE": "ICU_With",
            "ICU_RATE_WITH_DISEASE": "Rate_With_%",
            "TOTAL_WITHOUT_DISEASE": "N_Without",
            "ICU_YES_WITHOUT_DISEASE": "ICU_Without",
            "ICU_RATE_WITHOUT_DISEASE": "Rate_Without_%",
            "RATE_DIFFERENCE": "Diff_pp",
        }
    )
    # Keep Disease slightly wider so labels are not cramped.
    n_cols = len(q4_pdf.columns)
    disease_w = content_width * 0.125
    other_w = (content_width - disease_w) / (n_cols - 1)
    q4_col_widths = [disease_w] + [other_w] * (n_cols - 1)
    elements.append(table_from_df_custom_widths(q4_pdf, col_widths=q4_col_widths, font_size=7))
    elements.append(PageBreak())

    elements.append(Paragraph("Common diseases among deceased patients", styles["Heading3"]))
    elements.append(left_image(chart_deceased(deceased_df), content_width, 255))
    elements.append(Spacer(1, 6))
    elements.append(table_from_df(deceased_df))

    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()


render_logo(220)
st.title("Consolidated Report")

try:
    df = load_preprocessed_data()

    st.sidebar.header("Report Filters")
    outcomes = ["ALL", "POSITIVE", "NEGATIVE", "PENDING"]
    if "OUTCOME" in df.columns:
        observed_outcomes = sorted(df["OUTCOME"].dropna().astype(str).str.upper().unique().tolist())
        outcomes.extend([o for o in observed_outcomes if o not in outcomes])

    selected_outcome = st.sidebar.selectbox("Outcome", outcomes, key="report_outcome")
    min_age = int(df["AGE_NUM"].min(skipna=True))
    max_age = int(df["AGE_NUM"].max(skipna=True))
    selected_age = st.sidebar.slider("Age Range", min_age, max_age, (min_age, max_age), key="report_age")
    disease_options = [d for d in DISEASE_COLUMNS if d in df.columns]

    filtered = filter_dataset(df, selected_outcome, selected_age)

    age_counts = (
        filtered["AGE_GROUP"]
        .value_counts(sort=False)
        .reindex(AGE_LABELS, fill_value=0)
        .rename_axis("AGE_GROUP")
        .reset_index(name="COUNT")
    )
    gender_age = pd.crosstab(filtered["AGE_GROUP"], filtered["SEX"]).reindex(AGE_LABELS, fill_value=0)
    intub_counts = filtered["INTUBATED"].astype(str).str.upper().value_counts().reset_index()
    intub_counts.columns = ["INTUBATED", "COUNT"]
    intub_order = ["YES", "NO", "DOES NOT APPLY", "UNKNOWN", "IGNORED"]
    intub_counts["ORDER"] = intub_counts["INTUBATED"].apply(
        lambda x: intub_order.index(x) if x in intub_order else len(intub_order)
    )
    intub_counts = intub_counts.sort_values("ORDER").drop(columns=["ORDER"])
    intub_yes = int((filtered["INTUBATED"].astype(str).str.upper() == "YES").sum())

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
        icu_rate_without = (icu_yes_without_disease / total_without_disease) if total_without_disease > 0 else 0.0
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

    deceased = filtered[filtered["DATE_OF_DEATH"].notna()]
    deceased_rows = []
    for d in disease_options:
        deceased_rows.append({"DISEASE": d, "COUNT": int((deceased[d].astype(str).str.upper() == "YES").sum())})
    deceased_df = pd.DataFrame(deceased_rows).sort_values("COUNT", ascending=False)

    top_age = age_counts.sort_values("COUNT", ascending=False).head(1)["AGE_GROUP"].iloc[0] if len(age_counts) else "N/A"
    top_rate_diff = q4_table.head(1)
    top_rate_diff_text = (
        f"{top_rate_diff['DISEASE'].iloc[0]} ({top_rate_diff['RATE_DIFFERENCE'].iloc[0] * 100:.2f} pp)"
        if len(top_rate_diff)
        else "N/A"
    )
    top_deceased = deceased_df.head(1)
    top_deceased_text = f"{top_deceased['DISEASE'].iloc[0]} ({int(top_deceased['COUNT'].iloc[0])})" if len(top_deceased) else "N/A"

    executive_summary = [
        f"Filtered records analyzed: {len(filtered):,}.",
        f"Most represented age group: {top_age}.",
        f"Patients Required Intubation: {intub_yes:,}.",
        f"Largest ICU rate gap (with vs without disease): {top_rate_diff_text}.",
        f"Most common disease among deceased patients: {top_deceased_text}.",
    ]

    st.subheader("Executive Summary")
    for point in executive_summary:
        st.write(f"- {point}")

    st.divider()
    st.subheader("Age group susceptibility to COVID-19")
    st.plotly_chart(px.bar(age_counts, x="AGE_GROUP", y="COUNT", title="Cases by Age Group"), width="stretch")
    st.dataframe(age_counts, width="stretch")

    st.subheader("Distribution of cases by gender and age group")
    st.plotly_chart(px.imshow(gender_age, text_auto=True, aspect="auto", title="Heatmap: Age Group vs Gender"), width="stretch")
    st.dataframe(gender_age, width="stretch")

    st.subheader("Patients required intubation")
    st.metric("Patients Required Intubation", f"{intub_yes:,}")
    st.plotly_chart(
        px.area(
            intub_counts,
            x="INTUBATED",
            y="COUNT",
            markers=True,
            title="Intubation Distribution (Area Chart)",
        ).update_layout(height=420),
        width="stretch",
    )
    st.dataframe(intub_counts, width="stretch")

    st.subheader("Correlation Between Diseases and ICU Admission")
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
    fig_case4 = px.bar(
        q4_plot.assign(ICU_RATE_PCT=q4_plot["ICU_RATE"] * 100),
        x="DISEASE",
        y="ICU_RATE_PCT",
        color="GROUP",
        barmode="group",
        title="ICU Admission Rate by Disease (With vs Without Disease)",
        labels={"DISEASE": "Disease", "ICU_RATE_PCT": "ICU Rate (%)", "GROUP": "Group"},
    ).update_layout(height=460)
    st.plotly_chart(fig_case4, width="stretch")
    st.dataframe(q4_table, width="stretch")

    st.subheader("Common diseases among deceased patients")
    st.metric("Deceased records", f"{len(deceased):,}")
    st.plotly_chart(
        px.bar(deceased_df, x="COUNT", y="DISEASE", orientation="h", title="Common Diseases in Deceased Patients"),
        width="stretch",
    )
    st.dataframe(deceased_df, width="stretch")

    st.divider()
    st.subheader("Download Consolidated Report (PDF)")
    pdf_bytes = build_pdf_report(
        exec_summary=executive_summary,
        age_counts=age_counts,
        gender_age=gender_age,
        intub_counts=intub_counts,
        intub_yes=intub_yes,
        q4_table=q4_table,
        deceased_df=deceased_df,
    )
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.download_button(
        label="Download Consolidated PDF",
        data=pdf_bytes,
        file_name=f"consolidated_report_{ts}.pdf",
        mime="application/pdf",
    )

except Exception as exc:
    st.error(f"Consolidated report page failed: {exc}")
