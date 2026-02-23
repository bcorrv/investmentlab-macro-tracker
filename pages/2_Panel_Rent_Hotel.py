from __future__ import annotations

import streamlit as st
import pandas as pd

from utils_data import load_snapshot, load_run_summary


# -------------------------
# CONFIG UI
# -------------------------
st.set_page_config(page_title="Rent + Hotel", layout="wide", initial_sidebar_state="expanded")
st.title("🏢 Panel Inmobiliario — Renting + Hotel")
st.caption("Marco práctico: régimen macro → hurdle proxy → implicancias para renta y hotelería.")


# -------------------------
# LOAD DATA
# -------------------------
df = load_snapshot()
summary = load_run_summary()

df = df.copy()
df["fecha"] = pd.to_datetime(df["fecha"])
df = df.sort_values("fecha")

required_cols = ["usd_clp", "uf", "tpm", "dxy"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"Faltan columnas en daily_snapshot.csv: {missing}. Revisa main.py para incluirlas.")
    st.stop()

last = df.iloc[-1]
last_date = last["fecha"].date()

usd = float(last["usd_clp"])
uf = float(last["uf"])
tpm = float(last["tpm"])
dxy = float(last["dxy"])


# -------------------------
# SIDEBAR: ASSUMPTIONS
# -------------------------
with st.sidebar:
    st.subheader("Supuestos (ajustables)")

    spread_renting = st.slider("Spread sobre TPM (Renting)", 1.0, 8.0, 3.0, 0.25)
    spread_hotel = st.slider("Spread sobre TPM (Hotel)", 2.0, 10.0, 5.0, 0.25)

    st.divider()

    st.subheader("Targets (para lectura)")
    cap_rate_target = st.slider("Cap Rate target (Renting)", 4.0, 12.0, 7.0, 0.25)
    hotel_ebitda_margin = st.slider("Margen EBITDA Hotel target", 10, 45, 28, 1)

    st.divider()
    st.caption("Tip: estos targets no son “verdad”, son tu referencia para decidir y comparar.")


# -------------------------
# KPI HEADER
# -------------------------
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Fecha", str(last_date))
c2.metric("USD/CLP", f"{usd:,.2f}")
c3.metric("UF", f"{uf:,.2f}")
c4.metric("TPM", f"{tpm:.2f}%")
c5.metric("DXY", f"{dxy:.2f}")

if summary:
    st.caption(f"Último run (UTC): {summary.get('pipeline_run_utc', '—')} | versión: {summary.get('pipeline_version', '—')}")


# -------------------------
# SIGNALS / REGIME
# -------------------------
# USD vs MA30
tmp = df.dropna(subset=["usd_clp"]).copy()
tmp["usd_ma30"] = tmp["usd_clp"].rolling(30).mean()
usd_ma30 = float(tmp.iloc[-1]["usd_ma30"]) if len(tmp) >= 30 else None
usd_above_ma30 = bool(usd_ma30 is not None and usd > usd_ma30)

# DXY momentum (20d)
tmp2 = df.dropna(subset=["dxy"]).copy()
tmp2["dxy_ret20"] = tmp2["dxy"].pct_change(20) * 100
dxy_ret20 = float(tmp2.iloc[-1]["dxy_ret20"]) if len(tmp2) >= 21 else 0.0

# TPM level heuristic
tpm_high = tpm >= 6.0
tpm_mid = 4.0 <= tpm < 6.0

# Define regime (simple & explainable)
risk_off_signals = 0
risk_on_signals = 0

if not usd_above_ma30:
    risk_off_signals += 1
else:
    risk_on_signals += 1

if dxy_ret20 > 2.0:
    risk_off_signals += 1
elif dxy_ret20 < -2.0:
    risk_on_signals += 1

if tpm_high:
    risk_off_signals += 1
elif not tpm_mid:
    risk_on_signals += 1

if risk_off_signals >= 2:
    regime = "RISK-OFF"
    regime_color = "🔴"
elif risk_on_signals >= 2:
    regime = "RISK-ON"
    regime_color = "🟢"
else:
    regime = "NEUTRAL"
    regime_color = "🟡"


# -------------------------
# HURDLE PROXY
# -------------------------
hurdle_renting = tpm + spread_renting
hurdle_hotel = tpm + spread_hotel

st.divider()

r1, r2, r3 = st.columns([1, 1, 2])
r1.metric("Hurdle Proxy Renting", f"{hurdle_renting:.2f}%")
r2.metric("Hurdle Proxy Hotel", f"{hurdle_hotel:.2f}%")
r3.markdown(
    f"""
### {regime_color} Régimen: **{regime}**

**Señales:**
- USD vs MA30: **{"arriba" if usd_above_ma30 else "abajo"}** (MA30 ≈ {usd_ma30:,.2f} si disponible)
- DXY momentum (20d): **{dxy_ret20:.2f}%**
- TPM nivel: **{tpm:.2f}%**
"""
)

st.caption("Hurdle proxy = referencia rápida para disciplina (no reemplaza underwriting completo).")


# -------------------------
# RECOMMENDATIONS (actionable)
# -------------------------
st.subheader("Qué cambia según régimen (acciones concretas)")

colA, colB = st.columns(2)

with colA:
    st.markdown("### 🏘️ Renting (renta)")
    if regime == "RISK-OFF":
        st.write(
            "- Prioriza **calidad y liquidez** (ubicación prime, demanda estable).\n"
            "- Subir exigencia de **cap rate** / bajar precio objetivo.\n"
            "- **Más caja** y cláusulas defensivas (vacancia, reajustes, seguros).\n"
            "- No asumas compresión de tasas; modela escenario de tasas altas por más tiempo."
        )
    elif regime == "RISK-ON":
        st.write(
            "- Puedes tolerar **más duration** (compras con upside a mediano plazo).\n"
            "- Evalúa activos con **mejoras operativas** (reposición, optimización de renta).\n"
            "- Más sentido a proyectos con **reversión** y capex planificado.\n"
            "- Aun así: no bajes el hurdle por entusiasmo; compáralo contra tu target."
        )
    else:
        st.write(
            "- Mantén disciplina: **buen underwriting** y escenarios.\n"
            "- Foco en activos con **asimetría**: downside contenido, upside real.\n"
            "- Ajusta spreads con datos (banco, spreads reales, vacancia)."
        )

    st.caption(f"Referencia target cap rate: {cap_rate_target:.2f}% (solo guía).")

with colB:
    st.markdown("### 🏨 Hotel (inversión hotelera)")
    if regime == "RISK-OFF":
        st.write(
            "- Prioriza **resiliencia**: ADR defendible, demanda menos cíclica.\n"
            "- Exige **más margen** y más holgura de caja.\n"
            "- Reduce apalancamiento y fija tasas si es posible.\n"
            "- En expansiones: gate duro de permisos/capex; evita sobre-optimismo en ramp-up."
        )
    elif regime == "RISK-ON":
        st.write(
            "- Ventana para **capturar upside**: pricing power, demanda leisure.\n"
            "- Proyectos con capex bien controlado + narrativa comercial clara.\n"
            "- En operaciones: empuja RevPAR, paquetes, mix de canales.\n"
            "- Ojo: en hotel, el riesgo operativo no desaparece porque baje la tasa."
        )
    else:
        st.write(
            "- Mantén foco en **unidad económica**: margen, costos fijos, sensibilidad a ocupación.\n"
            "- Diseña estructura financiera con escenarios (ocupación, ADR, FX).\n"
            "- Decide por calidad de activos, no por “sentimiento macro”."
        )

    st.caption(f"Referencia target margen EBITDA hotel: {hotel_ebitda_margin}% (solo guía).")


# -------------------------
# CHARTS (simple, readable)
# -------------------------
st.divider()
st.subheader("Gráficos rápidos (contexto)")

g1, g2 = st.columns(2)

with g1:
    st.markdown("**USD/CLP + MA30 (últimos 180 días)**")
    d = df.tail(180).copy()
    d["usd_ma30"] = d["usd_clp"].rolling(30).mean()
    st.line_chart(d.set_index("fecha")[["usd_clp", "usd_ma30"]])

with g2:
    st.markdown("**DXY (últimos 180 días)**")
    d = df.tail(180).copy()
    st.line_chart(d.set_index("fecha")[["dxy"]])

st.divider()
with st.expander("Ver tabla (últimas 200 filas)"):
    st.dataframe(df.tail(200), use_container_width=True)