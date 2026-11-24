import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Benford Analysis Dashboard", layout="wide")


# ----------------------------------------------------------------------
# Benford + KS helper functions (NumPy only)
# ----------------------------------------------------------------------
def leading_digits(values, n_digits):
    """Return the first (1) or first two (2) digits of each non-zero value."""
    v = np.abs(np.asarray(values, dtype=float))
    v = v[np.isfinite(v) & (v > 0)]
    exponent = np.floor(np.log10(v))
    lead = np.floor(v / 10 ** (exponent - (n_digits - 1))).astype(int)
    # guard against floating point edge cases
    return np.clip(lead, 10 ** (n_digits - 1), 10 ** n_digits - 1)


def ks_test(actual_p, expected_p, n, alpha=0.05):
    """KS statistic = max absolute gap between the two cumulative distributions."""
    cdf_actual = np.cumsum(actual_p)
    cdf_expected = np.cumsum(expected_p)
    gaps = np.abs(cdf_actual - cdf_expected)
    d_stat = gaps.max()
    critical = np.sqrt(-0.5 * np.log(alpha / 2)) / np.sqrt(n)  # 1.36/sqrt(n) at 5%
    p_value = min(1.0, 2 * np.exp(-2 * n * d_stat ** 2))  # asymptotic approximation
    return d_stat, critical, p_value, cdf_actual, cdf_expected, int(gaps.argmax())


def mad_conformity(mad, n_digits):
    """Nigrini's mean absolute deviation conformity ranges."""
    cuts = (0.006, 0.012, 0.015) if n_digits == 1 else (0.0012, 0.0018, 0.0022)
    if mad <= cuts[0]:
        return "Close conformity"
    if mad <= cuts[1]:
        return "Acceptable conformity"
    if mad <= cuts[2]:
        return "Marginal conformity"
    return "Nonconformity"


def analyze(values, n_digits, alpha):
    lead = leading_digits(values, n_digits)
    n = len(lead)
    if n == 0:
        return None
    low = 10 ** (n_digits - 1)
    digits = np.arange(low, 10 ** n_digits)
    counts = np.bincount(lead - low, minlength=len(digits))
    actual_p = counts / n
    expected_p = np.log10(1 + 1 / digits)

    d_stat, critical, p_value, cdf_a, cdf_e, max_idx = ks_test(actual_p, expected_p, n, alpha)
    mad = np.mean(np.abs(actual_p - expected_p))

    table = pd.DataFrame(
        {
            "Digit": digits,
            "Count": counts,
            "Expected Count": expected_p * n,
            "Actual Proportion": actual_p,
            "Benford Proportion": expected_p,
            "Difference": actual_p - expected_p,
            "Actual CDF": cdf_a,
            "Benford CDF": cdf_e,
        }
    )
    summary = {
        "Test": "First digit" if n_digits == 1 else "First two digits",
        "N (non-zero values)": n,
        "KS statistic (D)": round(float(d_stat), 5),
        "Critical value": round(float(critical), 5),
        "Approx. p-value": round(float(p_value), 5),
        "Significance level": alpha,
        "Digit with largest CDF gap": int(digits[max_idx]),
        "MAD": round(float(mad), 5),
        "MAD conformity": mad_conformity(mad, n_digits),
        "Conclusion": (
            "Significantly DIFFERENT from Benford (reject H0)"
            if d_stat > critical
            else "NOT significantly different from Benford (fail to reject H0)"
        ),
    }
    return {"table": table, "summary": summary, "max_idx": max_idx, "n_digits": n_digits}


def make_plot(res, col):
    t = res["table"]
    n_digits = res["n_digits"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.bar(t["Digit"], t["Actual Proportion"], alpha=0.7, label="Actual")
    ax.plot(t["Digit"], t["Benford Proportion"], "r-o", markersize=3, label="Benford")
    ax.set_title(f"{'First digit' if n_digits == 1 else 'First two digits'}: actual vs Benford")
    ax.set_xlabel("Leading digit(s)")
    ax.set_ylabel("Proportion")
    if n_digits == 1:
        ax.set_xticks(t["Digit"])
    else:
        ax.set_xticks(range(10, 100, 10))
    ax.legend()

    ax = axes[1]
    ax.plot(t["Digit"], t["Actual CDF"], label="Actual CDF")
    ax.plot(t["Digit"], t["Benford CDF"], "r--", label="Benford CDF")
    x_max = t["Digit"].iloc[res["max_idx"]]
    ax.axvline(x_max, color="gray", linestyle=":", label=f"Max gap at {x_max}")
    ax.set_title("Cumulative distributions (KS statistic = max gap)")
    ax.set_xlabel("Leading digit(s)")
    ax.set_ylabel("Cumulative proportion")
    ax.legend()

    fig.suptitle(f"Column analyzed: {col}")
    fig.tight_layout()
    return fig


def fig_to_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    return buf.getvalue()


def to_csv_bytes(df, index=False):
    return df.to_csv(index=index).encode("utf-8")


# ----------------------------------------------------------------------
# App
# ----------------------------------------------------------------------
st.title("Benford's Law Analysis Dashboard")
st.write(
    "Upload a CSV or Excel file, filter it, and test whether a numeric column "
    "follows Benford's Law using the Kolmogorov-Smirnov (KS) statistic."
)

# ---- 1. Upload ----
uploaded = st.sidebar.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx", "xls"])
if uploaded is None:
    st.info("Upload a file in the sidebar to get started.")
    st.stop()

try:
    if uploaded.name.lower().endswith(".csv"):
        df = pd.read_csv(uploaded)
    else:
        sheets = pd.ExcelFile(uploaded).sheet_names
        sheet = st.sidebar.selectbox("Excel sheet", sheets)
        df = pd.read_excel(uploaded, sheet_name=sheet)
except Exception as e:
    st.error(f"Could not read the file: {e}")
    st.stop()

if df.empty:
    st.warning("The uploaded file has no rows.")
    st.stop()

# ---- 2. Filters (three criteria) ----
st.sidebar.header("Filters")
filtered = df.copy()

# Filter 1: categorical values
cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
f1_col = st.sidebar.selectbox("Filter 1 - category column", ["(none)"] + cat_cols)
if f1_col != "(none)":
    options = sorted(df[f1_col].dropna().astype(str).unique())
    chosen = st.sidebar.multiselect(f"Keep values in '{f1_col}'", options, default=options)
    filtered = filtered[filtered[f1_col].astype(str).isin(chosen)]

# Filter 2: numeric range
num_cols = df.select_dtypes(include=np.number).columns.tolist()
f2_col = st.sidebar.selectbox("Filter 2 - numeric range column", ["(none)"] + num_cols)
if f2_col != "(none)":
    lo, hi = float(df[f2_col].min()), float(df[f2_col].max())
    if lo < hi:
        rng = st.sidebar.slider(f"Range for '{f2_col}'", lo, hi, (lo, hi))
        filtered = filtered[filtered[f2_col].between(rng[0], rng[1])]
    else:
        st.sidebar.caption("That column has only one value; range filter skipped.")

# Filter 3: text search
f3_col = st.sidebar.selectbox("Filter 3 - text search column", ["(none)"] + list(df.columns))
if f3_col != "(none)":
    text = st.sidebar.text_input(f"'{f3_col}' contains")
    if text:
        filtered = filtered[filtered[f3_col].astype(str).str.contains(text, case=False, na=False)]

st.sidebar.write(f"**Rows after filtering:** {len(filtered):,} of {len(df):,}")
if filtered.empty:
    st.warning("No rows match the current filters.")
    st.stop()

# ---- Column of interest + settings ----
if not num_cols:
    st.error("This file has no numeric columns to analyze.")
    st.stop()

st.sidebar.header("Analysis")
target = st.sidebar.selectbox("Column of interest (numeric)", num_cols)
alpha = st.sidebar.selectbox("Significance level (alpha)", [0.05, 0.01, 0.10], index=0)

values = filtered[target].dropna()
res1 = analyze(values, 1, alpha)
res2 = analyze(values, 2, alpha)
if res1 is None:
    st.error("The selected column has no non-zero numeric values after filtering.")
    st.stop()
if res1["summary"]["N (non-zero values)"] < 100:
    st.warning("Fewer than 100 values: Benford tests are unreliable on small samples.")

fig1 = make_plot(res1, target)
fig2 = make_plot(res2, target)

summary_stats = filtered[target].describe().to_frame(name=target)
summary_stats.loc["missing"] = filtered[target].isna().sum()
summary_stats.loc["skew"] = filtered[target].skew()
ks_results = pd.DataFrame([res1["summary"], res2["summary"]])

# ---- Tabs ----
tab_data, tab_d1, tab_d2, tab_dl = st.tabs(
    ["Data & summary", "First digit", "First two digits", "Download & share"]
)

with tab_data:
    st.subheader("Filtered data preview")
    st.dataframe(filtered.head(200), use_container_width=True)
    st.subheader(f"Summary statistics: {target}")
    st.dataframe(summary_stats, use_container_width=True)
    hist_fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(values, bins=40)
    ax.set_title(f"Distribution of {target}")
    ax.set_xlabel(target)
    ax.set_ylabel("Frequency")
    st.pyplot(hist_fig)

for tab, res, fig, label in [
    (tab_d1, res1, fig1, "First-digit"),
    (tab_d2, res2, fig2, "First-two-digit"),
]:
    with tab:
        s = res["summary"]
        st.subheader(f"{label} Benford analysis")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("KS statistic (D)", s["KS statistic (D)"])
        c2.metric("Critical value", s["Critical value"])
        c3.metric("Approx. p-value", s["Approx. p-value"])
        c4.metric("MAD", s["MAD"])
        if s["KS statistic (D)"] > s["Critical value"]:
            st.error(f"{s['Conclusion']} at alpha = {alpha}.")
        else:
            st.success(f"{s['Conclusion']} at alpha = {alpha}.")
        st.caption(f"MAD conformity (Nigrini): {s['MAD conformity']}")
        st.pyplot(fig)
        st.dataframe(res["table"], use_container_width=True)

with tab_dl:
    st.subheader("Statistical results")
    st.dataframe(ks_results, use_container_width=True)

    st.subheader("Downloads")
    c1, c2, c3 = st.columns(3)
    c1.download_button("Summary statistics (CSV)", to_csv_bytes(summary_stats, index=True),
                       "summary_statistics.csv", "text/csv")
    c2.download_button("KS test results (CSV)", to_csv_bytes(ks_results),
                       "ks_results.csv", "text/csv")
    c3.download_button("Filtered data (CSV)", to_csv_bytes(filtered),
                       "filtered_data.csv", "text/csv")

    c4, c5 = st.columns(2)
    c4.download_button("First-digit chart (PNG)", fig_to_png(fig1),
                       "benford_first_digit.png", "image/png")
    c5.download_button("First-two-digit chart (PNG)", fig_to_png(fig2),
                       "benford_first_two_digits.png", "image/png")

    excel_buf = io.BytesIO()
    with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
        summary_stats.to_excel(writer, sheet_name="Summary Stats")
        ks_results.to_excel(writer, sheet_name="KS Results", index=False)
        res1["table"].to_excel(writer, sheet_name="First Digit", index=False)
        res2["table"].to_excel(writer, sheet_name="First Two Digits", index=False)
        filtered.to_excel(writer, sheet_name="Filtered Data", index=False)
    st.download_button(
        "Full report (Excel workbook)",
        excel_buf.getvalue(),
        "benford_report.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
