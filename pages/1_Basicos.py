from __future__ import annotations

from datetime import date
import io

import certifi
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

OWNER = "bcorrv"
DATA_REPO = "investmentlab-data"
BRANCH = "main"
SNAPSHOT_URL = f"https://raw.githubusercontent.com/{OWNER}/{DATA_REPO}/{BRANCH}/daily_snapshot.csv"

st.set_page_config(page_title="Básicos — Macro Tracker", layout="wide")
st.title("Básicos — Macro Tracker")

@st.cache_data(ttl=300)
def _get_text(url: str) -> str:
    r = requests.get(url, timeout=30, verify=certifi.where())
    r.raise_for_status()
    return r.text

@st.cache_data(ttl=300)
def load_snapshot() -> pd.DataFrame:
    txt = _get_text(SNAPSHOT_URL)
    df = pd.read_csv(io.StringIO(txt))
    df.columns = [c.strip() for c in df.columns]
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df = df.dropna(subset=["fecha"]).sort_values("fecha")

    for col in ["usd_clp", "uf", "tpm", "dxy"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def clamp_date_range(min_d: date, max_d: date, start_default: date, end_default: date) -> tuple[date, date]:
    s = max(min_d, start_default)
    e = min(max_d, end_default)
    if s > e:
        s = min_d
        e = max_d
    return s, e

def line_chart(df: pd.DataFrame, y_cols: list[str], title: str, yaxis_title: str = "") -> go.Figure:
    fig = go.Figure()
    for col in y_cols:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df["fecha"], y=df[col], mode="lines", name=col))
    fig.update_layout(
        title=title,
        xaxis_title="Fecha",
        yaxis_title=yaxis_title,
        height=420,
        margin=dict(l=20, r=20, t=60, b=20),
        legend_title="Serie",
    )
    return fig

df = load_snapshot()

min_date = df["fecha"].min().date()
max_date = df["fecha"].max().date()
default_start, default_end = clamp_date_range(
    min_date, max_date,
    start_default=date(max(2018, min_date.year), 1, 1),
    end_default=max_date
)

start, end = st.date_input(
    "Rango",
    value=(default_start, default_end),
    min_value=min_date,
    max_value=max_date,
)

d = df[(df["fecha"].dt.date >= start) & (df["fecha"].dt.date <= end)].copy().sort_values("fecha")

# KPIs arriba
usd_last = float(d["usd_clp"].dropna().iloc[-1]) if "usd_clp" in d.columns and d["usd_clp"].dropna().size else None
uf_last  = float(d["uf"].dropna().iloc[-1]) if "uf" in d.columns and d["uf"].dropna().size else None
tpm_last = float(d["tpm"].dropna().iloc[-1]) if "tpm" in d.columns and d["tpm"].dropna().size else None
dxy_last = float(d["dxy"].dropna().iloc[-1]) if "dxy" in d.columns and d["dxy"].dropna().size else None

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Última fecha", str(d["fecha"].max().date()) if len(d) else "n/a")
c2.metric("USD/CLP", f"{usd_last:,.2f}" if usd_last is not None else "n/a")
c3.metric("UF", f"{uf_last:,.2f}" if uf_last is not None else "n/a")
c4.metric("TPM", f"{tpm_last:.2f}%" if tpm_last is not None else "n/a")
c5.metric("DXY", f"{dxy_last:.2f}" if dxy_last is not None else "n/a")

# Gráficos básicos
st.plotly_chart(line_chart(d, ["usd_clp"], "USD/CLP", "CLP por USD"), use_container_width=True)
st.plotly_chart(line_chart(d, ["uf"], "UF", "UF"), use_container_width=True)

# TPM + DXY juntos (útil para “clima”)
cols = []
if "tpm" in d.columns: cols.append("tpm")
if "dxy" in d.columns: cols.append("dxy")
if cols:
    st.plotly_chart(line_chart(d, cols, "TPM y DXY", ""), use_container_width=True)

with st.expander("Ver datos (últimas 200 filas del rango)"):
    st.dataframe(d.tail(200), use_container_width=True)