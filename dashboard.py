from __future__ import annotations

from datetime import date
import io

import certifi
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# =========================
# CONFIG
# =========================
OWNER = "bcorrv"
DATA_REPO = "investmentlab-data"
BRANCH = "main"

SNAPSHOT_URL = f"https://raw.githubusercontent.com/{OWNER}/{DATA_REPO}/{BRANCH}/daily_snapshot.csv"
RUN_SUMMARY_URL = f"https://raw.githubusercontent.com/{OWNER}/{DATA_REPO}/{BRANCH}/run_summary.json"

# “Hurdle” (umbral) para renta/hotel cuando no tienes cap rates / yields reales aún.
# Esto NO es un cap rate real. Es un proxy de decisión (costo de capital + prima).
HURDLE_SPREAD_RENTING = 3.0  # puntos porcentuales sobre TPM (ajustable)
HURDLE_SPREAD_HOTEL = 4.0    # hotel suele exigir prima mayor (ajustable)

# =========================
# UI
# =========================
st.set_page_config(page_title="Investment Lab — Panel Inmobiliario", layout="wide")
st.title("Investment Lab — Panel Inmobiliario (Renting + Hotel)")

# =========================
# DATA LOADERS (robustos para SSL)
# =========================
@st.cache_data(ttl=300)
def _get_text(url: str) -> str:
    r = requests.get(url, timeout=30, verify=certifi.where())
    r.raise_for_status()
    return r.text

@st.cache_data(ttl=300)
def load_snapshot() -> pd.DataFrame:
    txt = _get_text(SNAPSHOT_URL)
    df = pd.read_csv(io.StringIO(txt))

    # Normaliza
    df.columns = [c.strip() for c in df.columns]

    # Fecha
    if "fecha" not in df.columns:
        raise ValueError(f"El snapshot no trae columna 'fecha'. Columnas: {df.columns.tolist()}")

    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df = df.dropna(subset=["fecha"]).sort_values("fecha")

    # Columnas esperadas (si faltan, no rompemos: solo avisamos)
    for col in ["usd_clp", "uf", "tpm", "dxy"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

@st.cache_data(ttl=300)
def load_run_summary() -> dict | None:
    try:
        txt = _get_text(RUN_SUMMARY_URL)
        import json
        return json.loads(txt)
    except Exception:
        return None


# =========================
# HELPERS
# =========================
def clamp_date_range(min_d: date, max_d: date, start_default: date, end_default: date) -> tuple[date, date]:
    s = max(min_d, start_default)
    e = min(max_d, end_default)
    if s > e:
        s = min_d
        e = max_d
    return s, e

def add_base100(df: pd.DataFrame, col: str, out_col: str) -> pd.DataFrame:
    d = df.copy()
    if col not in d.columns:
        d[out_col] = np.nan
        return d
    base = d[col].dropna()
    if base.empty:
        d[out_col] = np.nan
        return d
    base_val = float(base.iloc[0])
    if base_val == 0:
        d[out_col] = np.nan
        return d
    d[out_col] = (d[col] / base_val) * 100.0
    return d

def pct_change(df: pd.DataFrame, col: str, out_col: str) -> pd.DataFrame:
    d = df.copy()
    if col not in d.columns:
        d[out_col] = np.nan
        return d
    d[out_col] = d[col].pct_change() * 100.0
    return d

def rolling_mean(df: pd.DataFrame, col: str, window: int, out_col: str) -> pd.DataFrame:
    d = df.copy()
    if col not in d.columns:
        d[out_col] = np.nan
        return d
    d[out_col] = d[col].rolling(window).mean()
    return d

def safe_last(df: pd.DataFrame, col: str) -> float | None:
    if col not in df.columns:
        return None
    s = df[col].dropna()
    return float(s.iloc[-1]) if not s.empty else None

def regime_label(tpm: float | None, dxy_base100_delta: float | None, usd_over_ma30: bool | None) -> tuple[str, str]:
    """
    Devuelve (titulo, descripcion) para “régimen financiero” con reglas simples.
    """
    if tpm is None:
        return ("Régimen: sin TPM", "No hay TPM disponible para clasificar.")
    # Regla simple: TPM alta + dólar global fuerte = estrés.
    # Umbrales razonables iniciales (ajustables):
    tpm_high = tpm >= 6.0
    dxy_strong = (dxy_base100_delta is not None) and (dxy_base100_delta >= 3.0)
    usd_stress = bool(usd_over_ma30) if usd_over_ma30 is not None else False

    if tpm_high and (dxy_strong or usd_stress):
        return ("Régimen: ESTRÉS", "Costo de capital alto + USD fuerte. Prioriza liquidez, cobertura FX, pricing conservador.")
    if (not tpm_high) and (not dxy_strong) and (not usd_stress):
        return ("Régimen: VENTANA", "Condiciones más benignas. Puedes evaluar timing de compra/financiamiento con mayor agresividad.")
    return ("Régimen: MIXTO", "Señales cruzadas. Decide por sub-mercado y estructura de deuda; evita apuestas binarias.")


def line_chart(df: pd.DataFrame, x: str, y_cols: list[str], title: str, yaxis_title: str = "") -> go.Figure:
    fig = go.Figure()
    for col in y_cols:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df[x], y=df[col], mode="lines", name=col))
    fig.update_layout(
        title=title,
        xaxis_title="Fecha",
        yaxis_title=yaxis_title,
        legend_title="Serie",
        height=420,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig

def dual_axis_chart(df: pd.DataFrame, x: str, left_col: str, right_col: str, title: str, left_title: str, right_title: str) -> go.Figure:
    fig = go.Figure()

    if left_col in df.columns:
        fig.add_trace(go.Scatter(x=df[x], y=df[left_col], name=left_col, mode="lines", yaxis="y1"))
    if right_col in df.columns:
        fig.add_trace(go.Scatter(x=df[x], y=df[right_col], name=right_col, mode="lines", yaxis="y2"))

    fig.update_layout(
        title=title,
        xaxis=dict(title="Fecha"),
        yaxis=dict(title=left_title),
        yaxis2=dict(title=right_title, overlaying="y", side="right"),
        height=420,
        margin=dict(l=20, r=20, t=60, b=20),
        legend_title="Serie",
    )
    return fig


# =========================
# LOAD DATA
# =========================
df = load_snapshot()
min_date = df["fecha"].min().date()
max_date = df["fecha"].max().date()

# Panel salud
run_summary = load_run_summary()
colA, colB = st.columns([2, 3])
with colA:
    st.caption(f"Datos: {SNAPSHOT_URL}")
with colB:
    if run_summary and "pipeline_run_utc" in run_summary:
        st.caption(f"Último run (UTC): {run_summary['pipeline_run_utc']} | versión: {run_summary.get('pipeline_version', 'n/a')}")
    else:
        st.caption("Run summary: no disponible (ok si aún no lo publicas).")

# Selector de rango
default_start, default_end = clamp_date_range(
    min_date, max_date,
    start_default=date(max(2018, min_date.year), 1, 1),
    end_default=max_date
)

start, end = st.date_input(
    "Rango de análisis",
    value=(default_start, default_end),
    min_value=min_date,
    max_value=max_date,
)

mask = (df["fecha"].dt.date >= start) & (df["fecha"].dt.date <= end)
d = df.loc[mask].copy().sort_values("fecha")

# Enriquecimientos
d = rolling_mean(d, "usd_clp", 30, "usd_ma30")
d["usd_above_ma30"] = np.where(d["usd_ma30"].notna() & d["usd_clp"].notna(), d["usd_clp"] > d["usd_ma30"], np.nan)
d = pct_change(d, "usd_clp", "usd_daily_pct")
d = pct_change(d, "uf", "uf_daily_pct")

d = add_base100(d, "usd_clp", "usd_base100")
d = add_base100(d, "dxy", "dxy_base100")

# Divergencia “local”: cuánto se mueve USD/CLP vs DXY
if "usd_base100" in d.columns and "dxy_base100" in d.columns:
    d["usd_minus_dxy_base100"] = d["usd_base100"] - d["dxy_base100"]
else:
    d["usd_minus_dxy_base100"] = np.nan

# Para “tendencia TPM” (60 días hábiles aprox)
if "tpm" in d.columns:
    d["tpm_change_60"] = d["tpm"] - d["tpm"].shift(60)
else:
    d["tpm_change_60"] = np.nan

# Hurdles proxy
if "tpm" in d.columns:
    d["hurdle_renting"] = d["tpm"] + HURDLE_SPREAD_RENTING
    d["hurdle_hotel"] = d["tpm"] + HURDLE_SPREAD_HOTEL
else:
    d["hurdle_renting"] = np.nan
    d["hurdle_hotel"] = np.nan

# =========================
# EXEC SUMMARY
# =========================
st.subheader("Resumen Ejecutivo (Renting + Hotel)")

usd_last = safe_last(d, "usd_clp")
uf_last = safe_last(d, "uf")
tpm_last = safe_last(d, "tpm")
dxy_last = safe_last(d, "dxy")
last_dataset_date = d["fecha"].max().date() if not d.empty else None

# DXY delta base100 (rango)
dxy_base100_delta = None
if "dxy_base100" in d.columns:
    s = d["dxy_base100"].dropna()
    if len(s) >= 2:
        dxy_base100_delta = float(s.iloc[-1] - s.iloc[0])

usd_over_ma30 = None
if "usd_above_ma30" in d.columns:
    s = d["usd_above_ma30"].dropna()
    usd_over_ma30 = bool(s.iloc[-1]) if not s.empty else None

reg_title, reg_desc = regime_label(tpm_last, dxy_base100_delta, usd_over_ma30)

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Última fecha", str(last_dataset_date) if last_dataset_date else "n/a")
m2.metric("USD/CLP", f"{usd_last:,.2f}" if usd_last is not None else "n/a")
m3.metric("UF", f"{uf_last:,.2f}" if uf_last is not None else "n/a")
m4.metric("TPM", f"{tpm_last:.2f}%" if tpm_last is not None else "n/a")
m5.metric("DXY", f"{dxy_last:.2f}" if dxy_last is not None else "n/a")

st.markdown(f"**{reg_title}** — {reg_desc}")

# =========================
# TABS: RENTING vs HOTEL
# =========================
tab1, tab2 = st.tabs(["Renting (Renta)", "Hotel (Inversión hotelera)"])

with tab1:
    st.subheader("Renting — Riesgo macro que mueve yields y demanda")

    # 1) USD vs DXY (base 100)
    c1, c2 = st.columns([2, 1])
    with c1:
        fig = line_chart(
            d,
            x="fecha",
            y_cols=["usd_base100", "dxy_base100"],
            title="USD/CLP vs DXY (Base 100 en inicio del rango)",
            yaxis_title="Índice (Base 100)",
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        # Divergencia local (proxy riesgo Chile / FX local)
        div_last = safe_last(d, "usd_minus_dxy_base100")
        shock_last = safe_last(d, "usd_daily_pct")
        above_ma = safe_last(d, "usd_above_ma30")

        st.write("**Lecturas rápidas**")
        if div_last is not None:
            st.metric("Divergencia USD−DXY (pts)", f"{div_last:.2f}")
        if shock_last is not None:
            st.metric("Shock diario USD/CLP (%)", f"{shock_last:.2f}%")
        if above_ma is not None:
            st.write(f"USD sobre MA30: **{bool(above_ma)}**")

        st.info(
            "Si USD/CLP sube mucho más que DXY, suele ser componente local (CLP / riesgo / tasas). "
            "Eso afecta expectativas y spreads en activos de renta."
        )

    # 2) TPM + “hurdle renting”
    fig2 = dual_axis_chart(
        d,
        x="fecha",
        left_col="tpm",
        right_col="hurdle_renting",
        title="Costo de capital (TPM) y umbral proxy para renta (TPM + spread)",
        left_title="TPM (%)",
        right_title="Hurdle proxy (%)",
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.caption(
        f"Hurdle proxy Renting = TPM + {HURDLE_SPREAD_RENTING:.1f}%. "
        "No es cap rate real: es un umbral inicial para disciplinar decisiones."
    )

    # 3) UF (nivel y variación)
    fig3 = line_chart(d, x="fecha", y_cols=["uf"], title="UF (nivel)", yaxis_title="UF")
    st.plotly_chart(fig3, use_container_width=True)

with tab2:
    st.subheader("Hotel — FX + costo de capital + estrés")

    # 1) Régimen “estrés” (TPM + DXY + USD/MA30)
    c1, c2 = st.columns([2, 1])
    with c1:
        fig = line_chart(
            d,
            x="fecha",
            y_cols=["dxy_base100", "usd_base100"],
            title="Dólar global vs FX local (Base 100)",
            yaxis_title="Índice (Base 100)",
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.write("**Checklist de riesgo hotelero**")
        if tpm_last is not None:
            st.metric("TPM actual", f"{tpm_last:.2f}%")
        if dxy_base100_delta is not None:
            st.metric("Δ DXY (base100, pts)", f"{dxy_base100_delta:.2f}")
        if usd_over_ma30 is not None:
            st.write(f"USD sobre MA30: **{usd_over_ma30}**")

        st.warning(
            "Hotel: FX pega por demanda internacional + costos (importaciones/insumos) "
            "+ estructura de deuda. Un régimen de estrés cambia timing y estructura."
        )

    # 2) TPM + hurdle hotel
    fig2 = dual_axis_chart(
        d,
        x="fecha",
        left_col="tpm",
        right_col="hurdle_hotel",
        title="Costo de capital (TPM) y umbral proxy Hotel (TPM + spread)",
        left_title="TPM (%)",
        right_title="Hurdle proxy (%)",
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.caption(
        f"Hurdle proxy Hotel = TPM + {HURDLE_SPREAD_HOTEL:.1f}%. "
        "Proxy para disciplinar underwriting mientras no tengas yields/financiamiento real."
    )

    # 3) “Estrés” visual: USD vs MA30
    if "usd_clp" in d.columns and "usd_ma30" in d.columns:
        fig3 = line_chart(
            d,
            x="fecha",
            y_cols=["usd_clp", "usd_ma30"],
            title="USD/CLP y MA30 (señal de estrés / tendencia)",
            yaxis_title="CLP por USD",
        )
        st.plotly_chart(fig3, use_container_width=True)

# =========================
# DATA PREVIEW
# =========================
with st.expander("Ver datos (preview)"):
    st.dataframe(d.tail(200), use_container_width=True)