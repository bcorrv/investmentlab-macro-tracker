# dashboard.py
from __future__ import annotations

from pathlib import Path
import datetime as dt
from io import StringIO

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import requests
import certifi

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Investment Lab — Macro Dashboard", layout="wide")

# Repo público de data (RAW)
DATA_URL = "https://raw.githubusercontent.com/bcorrv/investmentlab-data/main/daily_snapshot.csv"

INDICATORS = {
    "USD/CLP": {"col": "usd_clp", "unit": "CLP", "show_ma30": True},
    "UF": {"col": "uf", "unit": "CLP", "show_ma30": False},
    "TPM": {"col": "tpm", "unit": "%", "show_ma30": False},
    "DXY": {"col": "dxy", "unit": "index", "show_ma30": False},
}

PRESETS = {
    "Personalizado": None,
    "2018–2019 (pre-pandemia)": (dt.date(2018, 1, 2), dt.date(2019, 12, 31)),
    "2020 (shock)": (dt.date(2020, 1, 1), dt.date(2020, 12, 31)),
    "2021–2022 (normalización)": (dt.date(2021, 1, 1), dt.date(2022, 12, 31)),
    "2023–2024 (tasas altas)": (dt.date(2023, 1, 1), dt.date(2024, 12, 31)),
    "Últimos 12 meses": "last12m",
    "Últimos 90 días": "last90d",
}


# =========================
# HELPERS
# =========================
@st.cache_data(ttl=300)  # refresca cada 5 min
def load_snapshot() -> pd.DataFrame:
    # Descarga via requests + certifi (evita SSL issues de urllib en Python 3.14)
    r = requests.get(DATA_URL, timeout=30, verify=certifi.where())
    r.raise_for_status()
    df = pd.read_csv(StringIO(r.text))

    if "fecha" not in df.columns:
        st.error("El CSV no tiene columna 'fecha'.")
        st.stop()

    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df = df.dropna(subset=["fecha"]).sort_values("fecha")

    # normaliza numéricos
    for c in ["usd_clp", "uf", "tpm", "dxy", "usd_pct_change", "usd_ma30", "usd_above_ma30"]:
        if c in df.columns:
            # usd_above_ma30 podría ser boolean; si falla, lo dejamos
            if c == "usd_above_ma30":
                continue
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def fmt_num(x: float | None, unit: str) -> str:
    if x is None or pd.isna(x):
        return "—"
    if unit == "%":
        return f"{float(x):.2f}%"
    if unit == "CLP":
        s = f"{float(x):,.2f}"
        return s.replace(",", "X").replace(".", ",").replace("X", ".")
    return f"{float(x):.2f}"


def y_range_from_series(series: pd.Series) -> tuple[float, float]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return (0.0, 1.0)
    ymin = float(s.min())
    ymax = float(s.max())
    if ymin == ymax:
        pad = abs(ymin) * 0.02 if ymin != 0 else 1.0
        return (ymin - pad, ymax + pad)
    pad = (ymax - ymin) * 0.08
    return (ymin - pad, ymax + pad)


def compute_metrics(dff: pd.DataFrame, value_col: str) -> dict:
    s = pd.to_numeric(dff[value_col], errors="coerce").dropna()
    if len(s) == 0:
        return {"last": None, "avg": None, "min": None, "max": None, "pct_last": None, "vol": None}

    last_ = float(s.iloc[-1])
    avg_ = float(s.mean())
    min_ = float(s.min())
    max_ = float(s.max())

    pct = s.pct_change() * 100.0
    pct_last = float(pct.iloc[-1]) if pct.dropna().shape[0] > 0 else None
    vol = float(pct.dropna().std()) if pct.dropna().shape[0] > 2 else None

    return {"last": last_, "avg": avg_, "min": min_, "max": max_, "pct_last": pct_last, "vol": vol}


def plot_single(dff: pd.DataFrame, value_col: str, title: str, height: int = 520) -> go.Figure:
    y0, y1 = y_range_from_series(dff[value_col])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dff["fecha"], y=dff[value_col], mode="lines", name=title))
    fig.update_yaxes(range=[y0, y1])
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10))
    return fig


# =========================
# APP
# =========================
df = load_snapshot()

st.title("📊 Investment Lab — Macro Dashboard")
st.caption(f"Fuente: GitHub RAW | Dataset: {df['fecha'].min().date()} → {df['fecha'].max().date()} | Filas: {len(df)}")

# --- Selector indicador principal
selected = st.selectbox("Indicador principal", list(INDICATORS.keys()), index=0)
cfg = INDICATORS[selected]
value_col = cfg["col"]
unit = cfg["unit"]

if value_col not in df.columns:
    st.error(f"El dataset no tiene columna '{value_col}'. Revisa el CSV publicado.")
    st.stop()

min_date = df["fecha"].min().date()
max_date = df["fecha"].max().date()

# --- Presets + rango
preset_name = st.selectbox("Período", list(PRESETS.keys()), index=0)

if PRESETS[preset_name] == "last12m":
    start_default = max(min_date, (pd.to_datetime(max_date) - pd.Timedelta(days=365)).date())
    end_default = max_date
elif PRESETS[preset_name] == "last90d":
    start_default = max(min_date, (pd.to_datetime(max_date) - pd.Timedelta(days=90)).date())
    end_default = max_date
elif isinstance(PRESETS[preset_name], tuple):
    start_default, end_default = PRESETS[preset_name]
    start_default = max(min_date, start_default)
    end_default = min(max_date, end_default)
else:
    start_default, end_default = min_date, max_date

start, end = st.date_input(
    "Rango (ajustable)",
    value=(start_default, end_default),
    min_value=min_date,
    max_value=max_date,
)

if isinstance(start, (list, tuple)):
    start, end = start[0], start[1]

# --- Filtrado principal
dff = df[(df["fecha"].dt.date >= start) & (df["fecha"].dt.date <= end)].copy()
dff = dff.dropna(subset=[value_col]).sort_values("fecha")

if dff.empty:
    st.warning("No hay datos para el indicador en ese rango.")
    st.stop()

m = compute_metrics(dff, value_col)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric(selected, fmt_num(m["last"], unit))
c2.metric("% diario", "—" if m["pct_last"] is None else f"{m['pct_last']:.2f}%")
c3.metric("Promedio", fmt_num(m["avg"], unit))
c4.metric("Mín / Máx", f"{fmt_num(m['min'], unit)} / {fmt_num(m['max'], unit)}")
c5.metric("Vol (% diaria)", "—" if m["vol"] is None else f"{m['vol']:.2f}")

# Señal MA30 solo si existe
if cfg.get("show_ma30") and "usd_ma30" in dff.columns:
    last_usd = float(dff["usd_clp"].iloc[-1])
    last_ma = dff["usd_ma30"].iloc[-1]
    if pd.notna(last_ma):
        last_ma = float(last_ma)
        if last_usd > last_ma:
            st.info("Régimen técnico USD: **sobre MA30**")
        else:
            st.success("Régimen técnico USD: **bajo MA30**")

st.subheader(f"{selected} en el período")
st.plotly_chart(plot_single(dff, value_col, selected), use_container_width=True)

with st.expander("Tabla (últimas 200 filas)"):
    st.dataframe(dff.tail(200), use_container_width=True)

# =========================
# COMPARADOR
# =========================
st.divider()
st.subheader("🔀 Comparador de indicadores")

left, right = st.columns(2)
with left:
    ind1 = st.selectbox("Indicador 1", list(INDICATORS.keys()), index=0, key="cmp_ind1")
with right:
    ind2 = st.selectbox("Indicador 2", list(INDICATORS.keys()), index=1, key="cmp_ind2")

mode = st.radio(
    "Modo",
    ["Dos gráficos", "Combinado (2 ejes Y)", "Combinado (normalizado base 100)"],
    horizontal=True,
    key="cmp_mode",
)

col1 = INDICATORS[ind1]["col"]
col2 = INDICATORS[ind2]["col"]

missing = [c for c in [col1, col2] if c not in df.columns]
if missing:
    st.warning(f"No están estas columnas en el CSV publicado: {missing}.")
else:
    dcomp = df[(df["fecha"].dt.date >= start) & (df["fecha"].dt.date <= end)].copy()
    dcomp[col1] = pd.to_numeric(dcomp[col1], errors="coerce")
    dcomp[col2] = pd.to_numeric(dcomp[col2], errors="coerce")
    dcomp = dcomp.dropna(subset=["fecha", col1, col2]).sort_values("fecha")

    if dcomp.empty:
        st.warning("No hay datos comunes para ambos indicadores en este rango.")
    else:
        if mode == "Dos gráficos":
            cA, cB = st.columns(2)
            with cA:
                st.caption(ind1)
                st.plotly_chart(plot_single(dcomp, col1, ind1, height=420), use_container_width=True)
            with cB:
                st.caption(ind2)
                st.plotly_chart(plot_single(dcomp, col2, ind2, height=420), use_container_width=True)

        elif mode == "Combinado (2 ejes Y)":
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(x=dcomp["fecha"], y=dcomp[col1], mode="lines", name=ind1), secondary_y=False)
            fig.add_trace(go.Scatter(x=dcomp["fecha"], y=dcomp[col2], mode="lines", name=ind2), secondary_y=True)
            fig.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10))
            fig.update_yaxes(title_text=ind1, secondary_y=False)
            fig.update_yaxes(title_text=ind2, secondary_y=True)
            st.plotly_chart(fig, use_container_width=True)

        else:  # Normalizado base 100
            base1 = float(dcomp[col1].iloc[0])
            base2 = float(dcomp[col2].iloc[0])

            if base1 == 0 or base2 == 0:
                st.warning("No se puede normalizar porque un valor base es 0.")
            else:
                dcomp["idx1"] = (dcomp[col1] / base1) * 100.0
                dcomp["idx2"] = (dcomp[col2] / base2) * 100.0

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=dcomp["fecha"], y=dcomp["idx1"], mode="lines", name=f"{ind1} (base 100)"))
                fig.add_trace(go.Scatter(x=dcomp["fecha"], y=dcomp["idx2"], mode="lines", name=f"{ind2} (base 100)"))

                y0, y1 = y_range_from_series(pd.concat([dcomp["idx1"], dcomp["idx2"]], ignore_index=True))
                fig.update_yaxes(range=[y0, y1])
                fig.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig, use_container_width=True)