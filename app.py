"""PhoneVault Analytics - mobile price tier classification dashboard (Microsoft Fluent light theme)."""

import html
import json
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
SAMPLE_TEST_FILE = ROOT / "test_data.csv"

# Fluent 2 design tokens (light)
BRAND = "#0F6CBD"
BRAND_DARK = "#115EA3"
BRAND_TINT = "#EBF3FC"
FG1, FG2, FG3 = "#242424", "#424242", "#616161"
STROKE, STROKE_SUBTLE = "#E0E0E0", "#F0F0F0"
SUCCESS, SUCCESS_BG = "#107C10", "#F1FAF1"
DANGER, DANGER_BG = "#C50F1F", "#FDF3F4"
FONT = "Segoe UI"
TIER_COLORS = ["#0F6CBD", "#038387", "#8764B8", "#CA5010"]
METRIC_ORDER = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
METRIC_HELP = {
    "Accuracy": "Share of devices assigned to the correct price tier.",
    "AUC": "One-vs-rest macro ROC AUC from predicted probabilities.",
    "Precision": "Macro-averaged precision across the four tiers.",
    "Recall": "Macro-averaged recall across the four tiers.",
    "F1": "Macro-averaged harmonic mean of precision and recall.",
    "MCC": "Matthews correlation coefficient (-1 to 1).",
}

st.set_page_config(
    page_title="PhoneVault Analytics",
    page_icon=":material/insights:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    f"""
<style>
    html, body, [class*="css"] {{
        font-family: 'Segoe UI Variable Text', 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
        color: {FG1};
    }}
    [data-testid="stAppViewContainer"] {{ background: #F5F5F5; }}
    [data-testid="stHeader"] {{ background: transparent; }}
    .block-container {{ padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1440px; }}

    /* Sidebar as a Fluent navigation pane */
    [data-testid="stSidebar"] {{ background: #FFFFFF; border-right: 1px solid {STROKE}; }}
    [data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {{ padding-top: 1rem; }}
    .nav-brand {{ display: flex; align-items: center; gap: 10px; padding: 4px 0 16px; border-bottom: 1px solid {STROKE_SUBTLE}; margin-bottom: 16px; }}
    .nav-brand .mark {{ width: 32px; height: 32px; border-radius: 6px; background: {BRAND}; color: #FFFFFF;
        display: flex; align-items: center; justify-content: center; font-weight: 600; font-size: 13px; }}
    .nav-brand .name {{ font-size: 15px; font-weight: 600; color: {FG1}; line-height: 1.2; }}
    .nav-brand .sub {{ font-size: 12px; color: {FG3}; }}
    .nav-label {{ font-size: 12px; font-weight: 600; color: {FG3}; text-transform: uppercase; letter-spacing: .04em; margin: 18px 0 6px; }}
    .param-row {{ display: flex; justify-content: space-between; font-size: 13px; padding: 6px 0; border-bottom: 1px solid {STROKE_SUBTLE}; }}
    .param-row span:first-child {{ color: {FG3}; }}
    .param-row span:last-child {{ color: {FG1}; font-weight: 600; font-family: Consolas, monospace; }}

    /* Page header */
    .page-header {{ background: #FFFFFF; border: 1px solid {STROKE}; border-radius: 8px; padding: 20px 24px;
        box-shadow: 0 0 2px rgba(0,0,0,.12), 0 1px 2px rgba(0,0,0,.14); display: flex; justify-content: space-between;
        align-items: center; gap: 16px; flex-wrap: wrap; margin-bottom: 16px; }}
    .breadcrumb {{ font-size: 12px; color: {FG3}; margin-bottom: 4px; }}
    .page-header h1 {{ font-size: 24px; font-weight: 600; color: {FG1}; margin: 0; padding: 0; line-height: 1.3; }}
    .page-header p {{ font-size: 14px; color: {FG2}; margin: 4px 0 0; }}
    .chips {{ display: flex; gap: 8px; flex-wrap: wrap; }}
    .chip {{ font-size: 12px; font-weight: 600; padding: 4px 10px; border-radius: 999px; border: 1px solid {STROKE};
        background: #FAFAFA; color: {FG2}; white-space: nowrap; }}
    .chip.brand {{ background: {BRAND_TINT}; border-color: #B4D6FA; color: {BRAND_DARK}; }}
    .chip.ok {{ background: {SUCCESS_BG}; border-color: #9FD89F; color: #0E700E; }}

    /* KPI cards */
    .kpi {{ background: #FFFFFF; border: 1px solid {STROKE}; border-radius: 8px; padding: 16px 18px 14px;
        box-shadow: 0 0 2px rgba(0,0,0,.12), 0 1px 2px rgba(0,0,0,.14); height: 100%; }}
    .kpi .label {{ font-size: 12px; font-weight: 600; color: {FG3}; text-transform: uppercase; letter-spacing: .04em; }}
    .kpi .value {{ font-size: 28px; font-weight: 600; color: {FG1}; margin: 6px 0 2px; line-height: 1.2; }}
    .kpi .delta {{ font-size: 12px; font-weight: 600; }}
    .kpi .delta.up {{ color: #0E700E; }}
    .kpi .delta.down {{ color: {DANGER}; }}
    .kpi .delta.flat {{ color: {FG3}; }}
    .kpi .bar {{ height: 4px; background: {STROKE_SUBTLE}; border-radius: 2px; margin-top: 10px; overflow: hidden; }}
    .kpi .bar > div {{ height: 100%; background: {BRAND}; border-radius: 2px; }}

    /* Cards built from st.container(border=True, key="card-...") */
    [class*="st-key-card"] {{ background: #FFFFFF; border: 1px solid {STROKE} !important; border-radius: 8px;
        box-shadow: 0 0 2px rgba(0,0,0,.12), 0 1px 2px rgba(0,0,0,.14); padding: 4px 6px; }}
    .card-title {{ font-size: 16px; font-weight: 600; color: {FG1}; margin: 0; }}
    .card-sub {{ font-size: 13px; color: {FG3}; margin: 2px 0 8px; }}

    /* Message bars */
    .msgbar {{ display: flex; gap: 10px; align-items: flex-start; padding: 10px 14px; border-radius: 4px; font-size: 14px;
        margin-bottom: 16px; border: 1px solid; color: {FG1}; }}
    .msgbar.info {{ background: {BRAND_TINT}; border-color: #B4D6FA; }}
    .msgbar.success {{ background: {SUCCESS_BG}; border-color: #9FD89F; }}
    .msgbar b {{ font-weight: 600; }}

    /* Tabs as a Fluent pivot */
    [data-baseweb="tab-list"] {{ gap: 4px; border-bottom: 1px solid {STROKE}; }}
    [data-baseweb="tab"] {{ font-size: 14px; font-weight: 600; color: {FG2}; padding: 10px 14px; }}
    [data-baseweb="tab"][aria-selected="true"] {{ color: {FG1}; }}
    [data-baseweb="tab-highlight"] {{ background-color: {BRAND}; height: 3px; border-radius: 2px; }}

    .stButton > button, .stDownloadButton > button {{ font-weight: 600; border-radius: 4px; }}
    [data-testid="stButtonGroup"] button[aria-checked="true"],
    [data-testid="stButtonGroup"] button[aria-checked="true"] * {{ color: #0E4775 !important; font-weight: 600; }}
    footer {{ visibility: hidden; }}
</style>
""",
    unsafe_allow_html=True,
)

@alt.theme.register("fluent", enable=True)
def fluent_theme():
    return alt.theme.ThemeConfig({
        "config": {
            "font": FONT,
            "background": "#FFFFFF",
            "view": {"stroke": "transparent"},
            "axis": {
                "labelColor": FG2, "titleColor": FG2, "labelFontSize": 12, "titleFontSize": 12,
                "titleFontWeight": 600, "gridColor": STROKE_SUBTLE, "domainColor": STROKE, "tickColor": STROKE,
            },
            "legend": {"labelColor": FG2, "titleColor": FG2, "labelFontSize": 12, "titleFontSize": 12},
            "title": {"color": FG1, "fontSize": 14, "fontWeight": 600},
        }
    })


@st.cache_resource
def load_artifacts():
    info = json.loads((MODEL_DIR / "model_info.json").read_text())
    models = {name: joblib.load(MODEL_DIR / file) for name, file in info["models"].items()}
    return info, models


def compute_metrics(model, X, y):
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    labels = list(model.classes_)
    try:
        auc = roc_auc_score(y, y_proba, multi_class="ovr", average="macro", labels=labels)
    except ValueError:
        # AUC is undefined if the upload lacks some classes
        auc = float("nan")
    metrics = {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": auc,
        "Precision": precision_score(y, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y, y_pred, average="macro", zero_division=0),
        "F1": f1_score(y, y_pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y, y_pred),
    }
    return metrics, y_pred, y_proba


def validate_upload(df, features, target):
    if df.empty:
        return "The uploaded file has no rows."
    missing = [c for c in features + [target] if c not in df.columns]
    if missing:
        return f"Missing required columns: {', '.join(missing)}"
    if df[features + [target]].isna().any().any():
        return "Required columns contain empty values."
    non_numeric = [c for c in features if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        return f"These feature columns must be numeric: {', '.join(non_numeric)}"
    unknown = set(df[target].unique().tolist()) - {0, 1, 2, 3}
    if unknown:
        return f"Target '{target}' must contain only 0-3; found {sorted(unknown)}"
    return None


@st.cache_data(show_spinner="Evaluating models...")
def evaluate_all(data: pd.DataFrame):
    X, y = data[FEATURES], data[TARGET]
    results = {}
    for name, model in models.items():
        metrics, y_pred, y_proba = compute_metrics(model, X, y)
        results[name] = {"metrics": metrics, "y_pred": y_pred, "y_proba": y_proba}
    return results


def fmt(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    return f"{value:.4f}"


def card_header(title, subtitle=None):
    st.markdown(f"<p class='card-title'>{title}</p>", unsafe_allow_html=True)
    if subtitle:
        st.markdown(f"<p class='card-sub'>{subtitle}</p>", unsafe_allow_html=True)


info, models = load_artifacts()
FEATURES, TARGET, CLASS_NAMES = info["features"], info["target"], info["class_names"]
TIER_MAP = dict(enumerate(CLASS_NAMES))

# ---------- Navigation pane ----------
with st.sidebar:
    st.markdown(
        """
        <div class='nav-brand'>
            <div class='mark'>PV</div>
            <div><div class='name'>PhoneVault Analytics</div><div class='sub'>Price tier intelligence</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='nav-label'>Data source</div>", unsafe_allow_html=True)
    source = st.segmented_control(
        "Data source", ["Sample test set", "Upload CSV"], default="Sample test set",
        required=True, label_visibility="collapsed", width="stretch",
    )
    uploaded = None
    if source == "Upload CSV":
        uploaded = st.file_uploader("Upload a CSV file", type="csv", label_visibility="collapsed")
    st.download_button(
        "Download sample CSV", data=SAMPLE_TEST_FILE.read_bytes(), file_name="phonevault_sample.csv",
        mime="text/csv", icon=":material/download:", width="stretch",
    )

    st.markdown("<div class='nav-label'>Classification model</div>", unsafe_allow_html=True)
    model_name = st.selectbox("Classification model", list(models.keys()), label_visibility="collapsed")
    st.markdown("<div class='nav-label'>Tuned hyperparameters</div>", unsafe_allow_html=True)
    st.markdown(
        "".join(f"<div class='param-row'><span>{k}</span><span>{v}</span></div>" for k, v in info["best_params"][model_name].items()),
        unsafe_allow_html=True,
    )

    st.markdown("<div class='nav-label'>About</div>", unsafe_allow_html=True)
    st.caption(
        "Pre-trained scikit-learn pipelines classify phones into four price tiers from 20 hardware "
        "specifications. Models are loaded, never retrained, in this app."
    )

# ---------- Load data ----------
if source == "Upload CSV" and uploaded is None:
    data, data_label = None, "No file"
else:
    try:
        data = pd.read_csv(uploaded if uploaded is not None else SAMPLE_TEST_FILE)
        data_label = html.escape(uploaded.name) if uploaded is not None else "test_data.csv"
    except Exception as exc:
        data, data_label = None, "Invalid file"
        st.error(f"The file could not be read: {exc}", icon=":material/error:")
        st.stop()

# ---------- Page header ----------
chips = f"<span class='chip brand'>{model_name}</span>"
if data is not None:
    chips += f"<span class='chip'>{data_label}</span><span class='chip ok'>{len(data):,} records</span>"
st.markdown(
    f"""
    <div class='page-header'>
        <div>
            <div class='breadcrumb'>PhoneVault &nbsp;/&nbsp; Model evaluation</div>
            <h1>Mobile Price Tier Dashboard</h1>
            <p>Evaluate pre-trained classifiers that predict a phone's market price tier from its hardware specifications.</p>
        </div>
        <div class='chips'>{chips}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if data is None:
    st.markdown(
        "<div class='msgbar info'><span>ℹ</span><div><b>Upload a dataset to begin.</b> "
        "Choose a CSV file in the navigation pane, or switch to <b>Sample test set</b>.</div></div>",
        unsafe_allow_html=True,
    )
    with st.container(border=True, key="card-format"):
        card_header("Expected file format", "One row per device with 20 numeric feature columns and the target column.")
        st.dataframe(
            pd.DataFrame({"Column": FEATURES + [TARGET],
                          "Role": ["Feature"] * len(FEATURES) + ["Target (0-3)"]}),
            hide_index=True, height=300,
        )
    st.stop()

error = validate_upload(data, FEATURES, TARGET)
if error:
    st.error(f"Validation failed: {error}", icon=":material/error:")
    st.stop()

X, y = data[FEATURES], data[TARGET]
results = evaluate_all(data)
selected = results[model_name]
metrics, y_pred, y_proba = selected["metrics"], selected["y_pred"], selected["y_proba"]
table = pd.DataFrame({name: r["metrics"] for name, r in results.items()}).T[METRIC_ORDER]

# ---------- KPI row ----------
kpi_cols = st.columns(6, gap="small")
for col, name in zip(kpi_cols, METRIC_ORDER):
    value = metrics[name]
    others = table.drop(index=model_name)[name].mean()
    if np.isnan(value) or np.isnan(others):
        delta_html = "<span class='delta flat'>No comparison</span>"
    else:
        diff = value - others
        cls, arrow = ("up", "▲") if diff > 0.0005 else ("down", "▼") if diff < -0.0005 else ("flat", "■")
        delta_html = f"<span class='delta {cls}'>{arrow} {abs(diff):.4f} vs. other models</span>"
    bar = 0 if np.isnan(value) else max(0.0, min(1.0, value)) * 100
    with col:
        st.markdown(
            f"""<div class='kpi' title='{METRIC_HELP[name]}'>
                <div class='label'>{name}</div>
                <div class='value'>{fmt(value)}</div>
                {delta_html}
                <div class='bar'><div style='width:{bar:.1f}%'></div></div>
            </div>""",
            unsafe_allow_html=True,
        )

st.write("")
tab_overview, tab_compare, tab_pred, tab_data = st.tabs(
    ["Overview", "Model comparison", "Predictions", "Data explorer"]
)

# ---------- Overview ----------
with tab_overview:
    left, right = st.columns([1, 1], gap="medium")
    with left, st.container(border=True, key="card-cm"):
        card_header("Confusion matrix", f"Actual vs. predicted price tier · {model_name}")
        cm = confusion_matrix(y, y_pred, labels=[0, 1, 2, 3])
        cm_df = pd.DataFrame(
            [(CLASS_NAMES[i], CLASS_NAMES[j], int(cm[i, j])) for i in range(4) for j in range(4)],
            columns=["Actual", "Predicted", "Count"],
        )
        threshold = cm.max() / 2
        base = alt.Chart(cm_df).encode(
            x=alt.X("Predicted:N", sort=CLASS_NAMES, title="Predicted tier", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Actual:N", sort=CLASS_NAMES, title="Actual tier"),
        )
        heat = base.mark_rect(stroke="#FFFFFF", strokeWidth=2).encode(
            color=alt.Color("Count:Q", scale=alt.Scale(range=["#F5F9FD", BRAND_DARK]), legend=None),
            tooltip=["Actual", "Predicted", "Count"],
        )
        labels = base.mark_text(fontSize=15, fontWeight=600).encode(
            text="Count:Q",
            color=alt.condition(alt.datum.Count > threshold, alt.value("#FFFFFF"), alt.value(FG1)),
        )
        st.altair_chart((heat + labels).properties(height=360), width="stretch")

    with right, st.container(border=True, key="card-report"):
        card_header("Per-tier performance", "Precision, recall and F1 for each price tier")
        report = classification_report(
            y, y_pred, labels=[0, 1, 2, 3], target_names=CLASS_NAMES, output_dict=True, zero_division=0,
        )
        rep = pd.DataFrame(report).T.drop(index=["accuracy"], errors="ignore").reset_index(names="Tier")
        rep["support"] = rep["support"].astype(int)
        st.dataframe(
            rep,
            hide_index=True,
            row_height=35,
            height=(len(rep) + 1) * 35 + 3,
            column_config={
                "Tier": st.column_config.TextColumn("Tier", width=110),
                "precision": st.column_config.ProgressColumn("Precision", format="%.3f", min_value=0, max_value=1, width=125),
                "recall": st.column_config.ProgressColumn("Recall", format="%.3f", min_value=0, max_value=1, width=125),
                "f1-score": st.column_config.ProgressColumn("F1", format="%.3f", min_value=0, max_value=1, width=125),
                "support": st.column_config.NumberColumn("Support", format="%d", width=70),
            },
        )
        errors = int((y_pred != y.to_numpy()).sum())
        far = int((np.abs(y_pred - y.to_numpy()) > 1).sum())
        c1, c2, c3 = st.columns(3)
        c1.metric("Correct", f"{len(y) - errors:,}", border=True)
        c2.metric("Misclassified", f"{errors:,}", border=True)
        c3.metric("Off by 2+ tiers", f"{far:,}", border=True)

# ---------- Model comparison ----------
with tab_compare:
    best_model = table["F1"].idxmax()
    st.markdown(
        f"<div class='msgbar success'><span>✓</span><div><b>{best_model}</b> leads on macro F1 "
        f"({table.loc[best_model, 'F1']:.3f}) across {len(table)} models on this dataset.</div></div>",
        unsafe_allow_html=True,
    )
    with st.container(border=True, key="card-table"):
        card_header("Benchmark", "All saved models evaluated on the current dataset · best value per metric highlighted")
        styled = (
            table.style.format("{:.4f}", na_rep="—")
            .highlight_max(axis=0, props=f"background-color: {BRAND_TINT}; color: {BRAND_DARK}; font-weight: 600;")
        )
        st.dataframe(styled)

    with st.container(border=True, key="card-chart"):
        card_header("Metric comparison", "Select metrics to compare side by side")
        chosen = st.segmented_control(
            "Metrics", METRIC_ORDER, selection_mode="multi", default=["Accuracy", "F1", "MCC"],
            label_visibility="collapsed",
        )
        if chosen:
            long = table[chosen].reset_index(names="Model").melt("Model", var_name="Metric", value_name="Score")
            chart = (
                alt.Chart(long)
                .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
                .encode(
                    x=alt.X("Model:N", title=None, axis=alt.Axis(labelAngle=0, labelLimit=160)),
                    xOffset="Metric:N",
                    y=alt.Y("Score:Q", scale=alt.Scale(domain=[0, 1]), title="Score"),
                    color=alt.Color("Metric:N", sort=chosen, legend=alt.Legend(orient="top", title=None)),
                    tooltip=["Model", "Metric", alt.Tooltip("Score:Q", format=".4f")],
                )
                .properties(height=360)
            )
            st.altair_chart(chart, width="stretch")
        else:
            st.caption("Select at least one metric.")

# ---------- Predictions ----------
with tab_pred:
    preview = data.copy()
    preview.insert(0, "Actual tier", y.map(TIER_MAP))
    preview.insert(1, "Predicted tier", pd.Series(y_pred, index=data.index).map(TIER_MAP))
    preview.insert(2, "Confidence", y_proba.max(axis=1))
    preview.insert(0, "Result", np.where(y_pred == y.to_numpy(), "Correct", "Misclassified"))

    with st.container(border=True, key="card-preds"):
        head_l, head_r = st.columns([3, 2], vertical_alignment="center")
        with head_l:
            card_header("Prediction results", f"Row-level output from {model_name}")
        with head_r:
            view = st.segmented_control(
                "Show", ["All", "Correct", "Misclassified"], default="All", required=True,
                label_visibility="collapsed", width="stretch",
            )
        shown = preview if view == "All" else preview[preview["Result"] == view]
        st.dataframe(
            shown,
            hide_index=True,
            height=440,
            column_config={
                "Result": st.column_config.TextColumn("Result", width="small"),
                "Confidence": st.column_config.ProgressColumn("Confidence", format="percent", min_value=0, max_value=1),
            },
        )
        f1, f2 = st.columns([3, 1], vertical_alignment="center")
        f1.caption(f"Showing {len(shown):,} of {len(preview):,} rows")
        f2.download_button(
            "Export results", preview.drop(columns=["Actual tier"]).assign(**{TARGET: y}).to_csv(index=False),
            "phonevault_predictions.csv", "text/csv", type="primary", icon=":material/file_download:", width="stretch",
        )

# ---------- Data explorer ----------
with tab_data:
    left, right = st.columns([2, 3], gap="medium")
    with left, st.container(border=True, key="card-dist"):
        card_header("Tier distribution", "Number of devices per actual price tier")
        dist = y.map(TIER_MAP).value_counts().reindex(CLASS_NAMES, fill_value=0).reset_index()
        dist.columns = ["Tier", "Devices"]
        dist["Share"] = dist["Devices"] / dist["Devices"].sum()
        bars = alt.Chart(dist).encode(
            x=alt.X("Tier:N", sort=CLASS_NAMES, title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Devices:Q", title="Devices"),
        )
        dist_chart = bars.mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3, width={"band": 0.6}).encode(
            color=alt.Color("Tier:N", sort=CLASS_NAMES, scale=alt.Scale(range=TIER_COLORS), legend=None),
            tooltip=["Tier", "Devices", alt.Tooltip("Share:Q", format=".1%")],
        ) + bars.mark_text(dy=-8, fontSize=12, fontWeight=600, color=FG1).encode(text="Devices:Q")
        st.altair_chart(dist_chart.properties(height=320), width="stretch")

    with right, st.container(border=True, key="card-feature"):
        h1, h2 = st.columns([3, 2], vertical_alignment="center")
        with h1:
            card_header("Feature by tier", "Distribution of a specification across price tiers")
        with h2:
            feature = st.selectbox("Feature", FEATURES, index=FEATURES.index("ram"), label_visibility="collapsed")
        fdf = pd.DataFrame({"Tier": y.map(TIER_MAP), feature: X[feature]})
        box = (
            alt.Chart(fdf)
            .mark_boxplot(size=40, median={"color": "#FFFFFF"}, ticks={"size": 20}, rule={"color": FG3})
            .encode(
                x=alt.X("Tier:N", sort=CLASS_NAMES, title=None, axis=alt.Axis(labelAngle=0)),
                y=alt.Y(f"{feature}:Q", title=feature),
                color=alt.Color("Tier:N", sort=CLASS_NAMES, scale=alt.Scale(range=TIER_COLORS), legend=None),
            )
            .properties(height=320)
        )
        st.altair_chart(box, width="stretch")

    with st.container(border=True, key="card-stats"):
        card_header("Summary statistics", f"{len(data):,} rows · {len(FEATURES)} features")
        st.dataframe(X.describe().T.round(2), height=380)

    with st.container(border=True, key="card-raw"):
        card_header("Raw data", "First 100 rows of the current dataset")
        st.dataframe(data.head(100), hide_index=True, height=380)
