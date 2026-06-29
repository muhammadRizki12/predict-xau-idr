import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from typing import Optional

st.set_page_config(
    page_title="XAU/IDR Prediction Dashboard",
    page_icon="📈",
    layout="wide",
)

# ── Constants ─────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent
PRED_DIR_FEB  = BASE_DIR / "data" / "prediction" / "feb"
PRED_DIR_JUNE = BASE_DIR / "data" / "prediction" / "june"
GT_PATH       = BASE_DIR / "data" / "ground-truth" / "gt-3maret-29jun.csv"

MODELS_FEB = {
    "3 Bulan": "pred_feb_3m.csv",
    "6 Bulan": "pred_feb_6m.csv",
    "1 Tahun": "pred_feb_1y.csv",
    "3 Tahun": "pred_feb_3y.csv",
}

MODELS_JUNE = {
    "3 Bulan": "pred_june_3m.csv",
    "6 Bulan": "pred_june_6m.csv",
    "1 Tahun": "pred_june_1y.csv",
    "3 Tahun": "pred_june_3y.csv",
}

COLORS_FEB = {
    "3 Bulan": "#0400d8",
    "6 Bulan": "#f77f00",
    "1 Tahun": "#4cc9f0",
    "3 Tahun": "#e63946",
}

COLORS_JUNE = {
    "3 Bulan": "#80b918",
    "6 Bulan": "#c77dff",
    "1 Tahun": "#ffd166",
    "3 Tahun": "#06d6a0",
}

GT_COLOR = "#ffffff"

CHART_LAYOUT = dict(
    xaxis_title="Tanggal",
    yaxis_title="Harga Close (IDR)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    hovermode="x unified",
    height=500,
    template="plotly_dark",
    yaxis=dict(tickformat=",.0f"),
    margin=dict(l=10, r=10, t=70, b=10),
)

# ── Data loaders ──────────────────────────────────────────────────────────────
@st.cache_data
def load_pred(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    df.index.name = "date"
    return df


@st.cache_data
def load_ground_truth() -> pd.DataFrame:
    df = pd.read_csv(GT_PATH)
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
    for col in ["Price", "Open", "High", "Low"]:
        df[col] = df[col].astype(str).str.replace(",", "").astype(float)
    df = df.rename(columns={"Date": "date", "Price": "close",
                             "Open": "open", "High": "high", "Low": "low"})
    return df.set_index("date").sort_index()[["open", "high", "low", "close"]]

# ── Helpers ───────────────────────────────────────────────────────────────────
def calc_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    err = y_pred - y_true
    return {
        "MAE":     err.abs().mean(),
        "RMSE":    (err ** 2).mean() ** 0.5,
        "MAPE (%)": (err.abs() / y_true).mean() * 100,
        "n":       len(y_true),
    }


def highlight_best(s: pd.Series) -> list:
    return ["background-color: #1a4731; font-weight:bold" if v == s.min() else "" for v in s]


def make_chart(
    title: str,
    models: dict,
    pred_dir: Path,
    colors: dict,
    dash: str,
    gt_df: Optional[pd.DataFrame] = None,
) -> go.Figure:
    fig = go.Figure()
    if gt_df is not None:
        fig.add_trace(go.Scatter(
            x=gt_df.index, y=gt_df["close"],
            name="Ground Truth",
            line=dict(color=GT_COLOR, width=2.5),
            mode="lines",
        ))
    for name, filename in models.items():
        df = load_pred(str(pred_dir / filename))
        fig.add_trace(go.Scatter(
            x=df.index, y=df["close"],
            name=name,
            line=dict(color=colors[name], width=1.8, dash=dash),
            mode="lines",
        ))
    fig.update_layout(title=dict(text=title, font_size=15), **CHART_LAYOUT)
    return fig


def render_metrics(models: dict, pred_dir: Path, gt_df: pd.DataFrame) -> None:
    rows = []
    for name, filename in models.items():
        pred_df    = load_pred(str(pred_dir / filename))
        common_idx = pred_df.index.intersection(gt_df.index)
        if len(common_idx) == 0:
            continue
        m = calc_metrics(gt_df.loc[common_idx, "close"], pred_df.loc[common_idx, "close"])
        rows.append({"Model": name,
                     "MAE (IDR)": m["MAE"], "RMSE (IDR)": m["RMSE"], "MAPE (%)": m["MAPE (%)"]})

    if not rows:
        return

    metrics = pd.DataFrame(rows).set_index("Model")
    st.dataframe(
        metrics.style
        .apply(highlight_best, subset=["MAE (IDR)", "RMSE (IDR)", "MAPE (%)"])
        .format({"MAE (IDR)": "{:,.2f}", "RMSE (IDR)": "{:,.2f}", "MAPE (%)": "{:.4f}"}),
        use_container_width=True,
    )
    col1, col2, col3 = st.columns(3)
    for col, metric, fmt in [
        (col1, "MAE (IDR)",  "{:,.2f} IDR"),
        (col2, "RMSE (IDR)", "{:,.2f} IDR"),
        (col3, "MAPE (%)",   "{:.4f}%"),
    ]:
        best = metrics[metric].idxmin()
        label = "Best " + metric.split(" ")[0]
        col.metric(label, best, fmt.format(metrics.loc[best, metric]))


def build_close_table(models: dict, pred_dir: Path, gt_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    combined = pd.DataFrame({"Ground Truth": gt_df["close"]}) if gt_df is not None else pd.DataFrame()
    for name, filename in models.items():
        combined[name] = load_pred(str(pred_dir / filename))["close"]
    return combined

# ── Load ──────────────────────────────────────────────────────────────────────
gt_df = load_ground_truth()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📈 XAU/IDR — Prediksi vs Ground Truth")
st.caption(
    f"Ground truth: {gt_df.index[0].date()} – {gt_df.index[-1].date()} "
    f"({len(gt_df)} hari trading) · Harga Close (IDR)"
)

# ── Section: Base Februari ────────────────────────────────────────────────────
st.subheader("Prediksi Base Data Februari")
st.plotly_chart(
    make_chart("Close Price XAU/IDR — Ground Truth vs Prediksi",
               MODELS_FEB, PRED_DIR_FEB, COLORS_FEB, "dot", gt_df=gt_df),
    use_container_width=True,
)
st.subheader("📊 Evaluasi Metrik — Base Data Februari")
st.caption("Dihitung pada tanggal yang overlap antara prediksi (Feb) dan ground truth (Mar–Jun 2026).")
render_metrics(MODELS_FEB, PRED_DIR_FEB, gt_df)

st.divider()

# ── Section: Base Juni ────────────────────────────────────────────────────────
st.subheader("Prediksi Base Data Juni")
st.plotly_chart(
    make_chart("Close Price XAU/IDR",
               MODELS_JUNE, PRED_DIR_JUNE, COLORS_JUNE, "dash"),
    use_container_width=True,
)
st.subheader("📊 Evaluasi Metrik — Base Data Juni")
june_start = load_pred(str(PRED_DIR_JUNE / list(MODELS_JUNE.values())[0])).index[0].date()
if june_start > gt_df.index[-1].date():
    st.info(
        f"Prediksi Juni dimulai dari **{june_start}**, sedangkan ground truth berakhir "
        f"**{gt_df.index[-1].date()}**. Tidak ada overlap — metrik tidak dapat dihitung."
    )
else:
    render_metrics(MODELS_JUNE, PRED_DIR_JUNE, gt_df)

st.divider()

# ── Data Table Expander ───────────────────────────────────────────────────────
with st.expander("Lihat Data Prediksi & Ground Truth"):
    tab_feb, tab_june = st.tabs(["Base Data Februari", "Base Data Juni"])
    with tab_feb:
        st.dataframe(
            build_close_table(MODELS_FEB, PRED_DIR_FEB, gt_df).style.format("{:,.2f}"),
            use_container_width=True,
        )
    with tab_june:
        st.dataframe(
            build_close_table(MODELS_JUNE, PRED_DIR_JUNE).style.format("{:,.2f}"),
            use_container_width=True,
        )
