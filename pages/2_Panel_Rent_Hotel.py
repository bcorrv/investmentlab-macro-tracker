from __future__ import annotations

import math
from datetime import date

import pandas as pd
import streamlit as st

# Tu proyecto ya usa esto (lo estás llamando desde esta página)
from utils_data import load_snapshot, load_run_summary


# -----------------------------
# Helpers financieros (simples)
# -----------------------------
def npv(rate: float, cashflows: list[float]) -> float:
    """NPV con cashflows donde cashflows[0] es hoy (t=0). rate en decimal."""
    if rate <= -1:
        return float("nan")
    total = 0.0
    for t, cf in enumerate(cashflows):
        total += cf / ((1.0 + rate) ** t)
    return total

def irr(cashflows: list[float], guess: float = 0.12) -> float:
    """
    IRR por Newton-Raphson.
    Retorna tasa en decimal (0.15 = 15%).
    """
    # Evita casos imposibles
    if not cashflows or (max(cashflows) <= 0) or (min(cashflows) >= 0):
        return float("nan")

    r = guess
    for _ in range(200):
        # NPV y derivada
        f = 0.0
        df = 0.0
        for t, cf in enumerate(cashflows):
            denom = (1.0 + r) ** t
            f += cf / denom
            if t > 0:
                df -= t * cf / ((1.0 + r) ** (t + 1))

        if abs(df) < 1e-12:
            return float("nan")

        r_new = r - f / df

        # Convergencia
        if abs(r_new - r) < 1e-8:
            return r_new

        # Evita explotar
        if r_new <= -0.9999 or r_new > 10:
            return float("nan")

        r = r_new

    return float("nan")


def calc_wacc(cost_of_equity: float, cost_of_debt: float, debt_pct: float, tax_rate: float) -> float:
    """
    WACC = E/V*Re + D/V*Rd*(1-T)
    inputs en decimal (0.12=12%)
    """
    d = max(0.0, min(1.0, debt_pct))
    e = 1.0 - d
    return e * cost_of_equity + d * cost_of_debt * (1.0 - tax_rate)

def safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def fmt_pct(x: float) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    return f"{x*100:.2f}%"


def fmt_num(x: float) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    return f"{x:,.2f}"


# -----------------------------
# App
# -----------------------------
st.set_page_config(page_title="Panel Rent + Hotel", layout="wide")

st.title("🏢 Panel Inmobiliario — Renting + Hotel")
st.caption("Marco práctico: régimen macro → hurdle proxy → disciplina de inversión (no reemplaza underwriting completo).")

df = load_snapshot()
run_summary = load_run_summary()

# Normaliza tipos
df["fecha"] = pd.to_datetime(df["fecha"]).dt.date

# Info de run (si existe)
colA, colB = st.columns([2, 3])
with colA:
    st.write(f"**Datos:** {run_summary.get('data_url','(local)')}")
with colB:
    st.write(
        f"**Último run (UTC):** {run_summary.get('pipeline_run_utc','—')}  |  "
        f"**versión:** {run_summary.get('pipeline_version','—')}"
    )

# Rango global disponible
min_date: date = df["fecha"].min()
max_date: date = df["fecha"].max()

# Sidebar: supuestos
st.sidebar.header("Supuestos (ajustables)")

spread_rent = st.sidebar.slider("Spread sobre TPM (Renting)", 0.0, 8.0, 3.0, 0.25)
spread_hotel = st.sidebar.slider("Spread sobre TPM (Hotel)", 0.0, 12.0, 5.0, 0.25)

st.sidebar.divider()
st.sidebar.header("Targets (para lectura)")

cap_rate_target = st.sidebar.slider("Cap Rate target (Renting)", 3.0, 12.0, 7.0, 0.25)
margen_ebitda_target = st.sidebar.slider("Margen EBITDA Hotel target", 10, 60, 28, 1)

st.sidebar.caption("Tip: estos targets no son “verdad”, son tu referencia para decidir y comparar.")

# Selector rango
st.subheader("Rango de análisis")
start, end = st.date_input(
    "Rango (ajustable)",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

if isinstance(start, tuple) or isinstance(start, list):
    # Por si Streamlit devuelve tuplas (raro)
    start, end = start[0], start[1]

mask = (df["fecha"] >= start) & (df["fecha"] <= end)
d = df.loc[mask].copy()

if d.empty:
    st.error("El rango seleccionado no tiene datos.")
    st.stop()

# Última fila (estado actual)
last = d.iloc[-1]
last_date = last["fecha"]

usd = safe_float(last.get("usd_clp"))
uf = safe_float(last.get("uf"))
tpm = safe_float(last.get("tpm")) / 100.0 if safe_float(last.get("tpm")) > 1 else safe_float(last.get("tpm"))  # tolerante
dxy = safe_float(last.get("dxy"))

# MA30 (si existe en snapshot)
ma30 = safe_float(last.get("usd_ma30"), default=float("nan"))
usd_above = bool(last.get("usd_above_ma30")) if "usd_above_ma30" in last else (usd > ma30 if not math.isnan(ma30) else False)

# Momentum DXY (20 días) si hay data suficiente
dxy_mom_20 = None
if "dxy" in d.columns and d["dxy"].notna().sum() > 25:
    dd = d.dropna(subset=["dxy"]).copy()
    if len(dd) > 21:
        dxy_mom_20 = (dd["dxy"].iloc[-1] / dd["dxy"].iloc[-21] - 1.0)

# Hurdles proxy (TPM + spread)
hurdle_rent = tpm + (spread_rent / 100.0)
hurdle_hotel = tpm + (spread_hotel / 100.0)

# Régimen simple (ejemplo disciplinario)
signals = []
signals.append(f"USD vs MA30: {'arriba' if usd_above else 'abajo'} (MA30 ≈ {fmt_num(ma30)} si disponible)")
if dxy_mom_20 is not None:
    signals.append(f"DXY momentum (20d): {dxy_mom_20*100:.2f}%")
signals.append(f"TPM nivel: {tpm*100:.2f}%")

# regla simple para etiqueta de régimen
score = 0
score += 1 if usd_above else -1
if dxy_mom_20 is not None:
    score += 1 if dxy_mom_20 > 0 else -1

if score >= 2:
    regime = "RISK-ON (USD fuerte)"
elif score <= -2:
    regime = "RISK-OFF (USD débil / alivio)"
else:
    regime = "NEUTRAL"


# -----------------------------
# Resumen ejecutivo
# -----------------------------
st.subheader("Resumen Ejecutivo (Renting + Hotel)")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Fecha", str(last_date))
c2.metric("USD/CLP", f"{usd:,.2f}")
c3.metric("UF", f"{uf:,.2f}")
c4.metric("TPM", f"{tpm*100:.2f}%")
c5.metric("DXY", f"{dxy:,.2f}")

c6, c7, c8 = st.columns([1, 1, 2])
c6.metric("Hurdle Proxy Renting", f"{hurdle_rent*100:.2f}%")
c7.metric("Hurdle Proxy Hotel", f"{hurdle_hotel*100:.2f}%")

with c8:
    st.markdown(f"### 🟡 Régimen: **{regime}**")
    st.write("**Señales:**")
    for s in signals:
        st.write(f"- {s}")

st.caption("Hurdle proxy = TPM + spread (disciplina rápida). No reemplaza underwriting completo.")

# -----------------------------
# Tabs: Renting / Hotel + Simulador
# -----------------------------
tab_rent, tab_hotel, tab_sim = st.tabs(["Renting (Renta)", "Hotel (Inversión hotelera)", "Simulador (Decisión)"])


with tab_rent:
    st.markdown("## Renting — Riesgo macro que mueve yields y demanda")
    st.write(
        "Lectura: si sube costo de capital (TPM), suele presionar cap rates (yields). "
        "USD/CLP y DXY te ayudan a separar componente global vs local."
    )

    # Serie base 100: USD vs DXY
    if "dxy" in d.columns and d["dxy"].notna().any():
        base_usd = d["usd_clp"].iloc[0]
        base_dxy = d["dxy"].dropna().iloc[0] if d["dxy"].dropna().shape[0] > 0 else None

        plot_df = d.copy()
        plot_df["usd_base100"] = (plot_df["usd_clp"] / base_usd) * 100.0
        if base_dxy is not None:
            plot_df["dxy_base100"] = (plot_df["dxy"] / base_dxy) * 100.0

        st.line_chart(
            plot_df.set_index("fecha")[["usd_base100"] + (["dxy_base100"] if "dxy_base100" in plot_df else [])]
        )

    # TPM y hurdle
    if "tpm" in d.columns:
        tmp = d.copy()
        tmp["tpm_pct"] = tmp["tpm"]
        tmp["hurdle_rent_pct"] = tmp["tpm_pct"] + spread_rent
        st.line_chart(tmp.set_index("fecha")[["tpm_pct", "hurdle_rent_pct"]])

    st.info(
        "Idea práctica: cuando TPM sube, tu disciplina debería exigir **más yield/cap rate** o **mejor calidad**. "
        "Evita “cap rates mágicos” sin prima por riesgo."
    )


with tab_hotel:
    st.markdown("## Hotel — Inversión hotelera (unidad económica)")
    st.write(
        "Hotel no es sólo cap rate. Es: ocupación, ADR, margen, estacionalidad, FX y estructura de deuda. "
        "Aquí el hurdle proxy te fuerza disciplina antes del Excel pesado."
    )

    st.write(f"Referencia target margen EBITDA (solo guía): **{margen_ebitda_target}%**")
    st.write("Checklist rápido:")
    st.write("- ¿El FX (USD/CLP) te mejora costos o te los empeora?")
    st.write("- ¿El costo de deuda (TPM + spread) te mata el DSCR?")
    st.write("- ¿Tu upside depende de macro o de operación (ADR/ocupación/margen)?")

    # Quick charts
    left, right = st.columns(2)

    with left:
        st.caption("USD/CLP + MA30 (últimos 180 días del rango)")
        dd = d.tail(180).copy()
        cols = ["usd_clp"] + (["usd_ma30"] if "usd_ma30" in dd.columns else [])
        st.line_chart(dd.set_index("fecha")[cols])

    with right:
        st.caption("DXY (últimos 180 días del rango)")
        if "dxy" in d.columns:
            st.line_chart(d.tail(180).set_index("fecha")[["dxy"]])


with tab_sim:
    st.markdown("## Simulador simple — Decisión de inversión (Hotel)")
    st.caption("Modelo simple: deuda bullet (principal se paga al final), interés anual, cashflows anuales constantes.")

    # ---------
    # Inputs
    # ---------
    i1, i2, i3 = st.columns(3)

    with i1:
        capex = st.number_input("CAPEX total (USD)", min_value=0.0, value=10_000_000.0, step=250_000.0, format="%.2f")
        years = st.number_input("Horizonte (años)", min_value=3, max_value=20, value=10, step=1)

        st.subheader("Modelo Operativo (deriva EBITDA)")

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            rooms = st.number_input("Habitaciones", min_value=10, max_value=800, value=120, step=5)

        with c2:
            occ_pct = st.slider("Ocupación (%)", min_value=30.0, max_value=90.0, value=65.0, step=0.5)

        with c3:
            adr = st.number_input("ADR (USD)", min_value=50.0, max_value=3000.0, value=250.0, step=10.0)

        with c4:
            ebitda_margin_pct = st.slider("Margen EBITDA (%)", min_value=10.0, max_value=70.0, value=35.0, step=1.0)

        occ = occ_pct / 100.0
        ebitda_margin = ebitda_margin_pct / 100.0

        revpar = adr * occ
        revenue = revpar * rooms * 365.0
        ebitda = revenue * ebitda_margin

        st.caption("EBITDA = Habitaciones × Ocupación × ADR × 365 × Margen EBITDA")
        st.metric("EBITDA derivado (USD/año)", f"{ebitda:,.0f}")

    with i2:
        debt_pct = st.slider("% Deuda", 0, 80, 55, 5)
        spread_debt = st.slider("Spread deuda sobre TPM (bps)", 0, 900, 400, 25)  # 400 bps = 4.00%
        tax = st.slider("Impuesto efectivo (proxy) (%)", 0.0, 35.0, 0.0, 1.0)

    with i3:
        cost_equity = st.slider("Costo de Equity (Re) (%)", 6.0, 30.0, 16.0, 0.25)
        discount = st.slider("Discount rate (hurdle) (%)", 0.0, 25.0, float(hurdle_hotel * 100.0), 0.25)

        exit_method = st.selectbox("Salida", ["Multiple EBITDA", "Cap rate sobre EBITDA"], index=0)
        if exit_method == "Multiple EBITDA":
            exit_multiple = st.slider("Exit multiple (x EBITDA)", 4.0, 20.0, 10.0, 0.25)
            exit_cap = None
        else:
            exit_cap = st.slider("Exit cap rate (%)", 4.0, 15.0, 9.0, 0.25)
            exit_multiple = None

    # -----------------------
    # Cálculos (orden fijo)
    # -----------------------
    debt = capex * (debt_pct / 100.0)
    equity = capex - debt

    # Tasas (en decimal)
    tax_rate = tax / 100.0
    disc_rate = discount / 100.0
    re = cost_equity / 100.0

    # debt_rate SIEMPRE se define aquí (no puede faltar)
    debt_rate = tpm + (spread_debt / 10_000.0)  # bps -> decimal
    rd = debt_rate
    d_pct = debt_pct / 100.0

    wacc = calc_wacc(re, rd, d_pct, tax_rate)

    # Flujo anual equity (simplificado)
    ebitda_after_tax = ebitda * (1.0 - tax_rate)
    interest = debt * debt_rate
    cf_annual = ebitda_after_tax - interest

    # DSCR proxy (EBITDA / interés)
    dscr = (ebitda / interest) if interest > 0 else float("inf")

    # Exit value
    if exit_method == "Multiple EBITDA":
        exit_value = ebitda * exit_multiple
    else:
        exit_value = ebitda / (exit_cap / 100.0) if exit_cap and exit_cap > 0 else float("nan")

    exit_equity = exit_value - debt  # paga principal al final

    # Cashflows Equity
    cashflows = [-equity] + [cf_annual] * (int(years) - 1) + [cf_annual + exit_equity]

    # Unlevered cashflows (sin deuda)
    cashflows_unlev = [-capex] + [ebitda_after_tax] * (int(years) - 1) + [ebitda_after_tax + exit_value]

    # Métricas
    npv_equity = npv(disc_rate, cashflows)
    irr_equity = irr(cashflows, guess=max(disc_rate, 0.12) if disc_rate > 0 else 0.12)

    irr_unlev = irr(cashflows_unlev, guess=max(disc_rate, 0.12) if disc_rate > 0 else 0.12)

    unlevered_yield = (ebitda / capex) if capex > 0 else float("nan")
    cash_yield = (cf_annual / equity) if equity > 0 else float("nan")

    # Sensibilidades ±100 bps (discount)
    npv_disc_up = npv(disc_rate + 0.01, cashflows)
    npv_disc_dn = npv(max(disc_rate - 0.01, -0.99), cashflows)

    # Sensibilidad deuda +100 bps
    cf_annual_up_debt = ebitda_after_tax - (debt * (debt_rate + 0.01))
    cashflows_up_debt = [-equity] + [cf_annual_up_debt] * (int(years) - 1) + [cf_annual_up_debt + exit_equity]
    npv_debt_up = npv(disc_rate, cashflows_up_debt)

    # -------------
    # Output
    # -------------
    o1, o2, o3, o4 = st.columns(4)
    o1.metric("Equity (USD)", fmt_num(equity))
    o2.metric("Deuda (USD)", fmt_num(debt))
    o3.metric("Tasa deuda (TPM+spread)", f"{debt_rate*100:.2f}%")
    o4.metric("WACC (proxy)", f"{wacc*100:.2f}%")

    o5, o6, o7, o8 = st.columns(4)
    o5.metric("Discount (hurdle)", f"{disc_rate*100:.2f}%")
    o6.metric("IRR Equity", "—" if math.isnan(irr_equity) else f"{irr_equity*100:.2f}%")
    o7.metric("IRR Unlevered", "—" if math.isnan(irr_unlev) else f"{irr_unlev*100:.2f}%")
    o8.metric("DSCR (EBITDA/Interés)", f"{dscr:.2f}x" if math.isfinite(dscr) else "∞")

    st.divider()

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("NPV Equity (USD)", fmt_num(npv_equity))
    r2.metric("CF anual equity (USD)", fmt_num(cf_annual))
    r3.metric("Unlevered yield (EBITDA/CAPEX)", fmt_pct(unlevered_yield))
    r4.metric("Cash yield equity (CF/Equity)", fmt_pct(cash_yield))

    st.write("**Sensibilidades (NPV equity):**")
    st.write(f"- Discount +100 bps: **{fmt_num(npv_disc_up)}**")
    st.write(f"- Discount -100 bps: **{fmt_num(npv_disc_dn)}**")
    st.write(f"- Deuda +100 bps: **{fmt_num(npv_debt_up)}**")

    st.divider()
    st.subheader("Lectura rápida (disciplina)")

    bullets = []
    if npv_equity > 0:
        bullets.append("✅ NPV equity positivo al hurdle elegido → pasa primer filtro.")
    else:
        bullets.append("🛑 NPV equity negativo al hurdle → no compensa riesgo (o ajusta precio/operación/estructura).")

    if dscr < 1.3:
        bullets.append("⚠️ DSCR bajo (<1.3x) → deuda agresiva para hotel (sube equity o baja tasa).")
    else:
        bullets.append("✅ DSCR razonable (≥1.3x) → deuda más defendible.")

    if cash_yield < disc_rate:
        bullets.append("⚠️ Cash yield < hurdle → dependes demasiado del exit/optimismo.")
    else:
        bullets.append("✅ Cash yield ≥ hurdle → retorno más sólido sin “rezar por el exit”.")

    for b in bullets:
        st.write(f"- {b}")

    st.caption("Modelo intencionalmente simple. Sirve para disciplina, no reemplaza underwriting completo.")

    st.divider()
    st.subheader("Mapa de Sensibilidad — NPV Equity")

    import numpy as np
    import pandas as pd
    import plotly.express as px

    # Rango de EBITDA ±20%
    ebitda_range = np.linspace(ebitda * 0.8, ebitda * 1.2, 10)

    # Rango de exit cap ±150 bps si es cap rate
    if exit_method == "Cap rate sobre EBITDA":
        exit_range = np.linspace((exit_cap - 1.5), (exit_cap + 1.5), 10)
    else:
        exit_range = np.linspace((exit_multiple - 2), (exit_multiple + 2), 10)

    heat_data = []

    for e in ebitda_range:
        for ex in exit_range:

            # recalcular exit value
            if exit_method == "Cap rate sobre EBITDA":
                ev = e / (ex / 100.0)
            else:
                ev = e * ex

            exit_eq = ev - debt
            cf = e * (1 - tax_rate) - (debt * debt_rate)

            cashflows_temp = [-equity] + [cf] * (int(years) - 1) + [cf + exit_eq]
            npv_temp = npv(disc_rate, cashflows_temp)

            heat_data.append({
                "EBITDA": e,
                "Exit": ex,
                "NPV": npv_temp
            })

    heat_df = pd.DataFrame(heat_data)

    pivot = heat_df.pivot(index="EBITDA", columns="Exit", values="NPV")

    fig = px.imshow(
        pivot,
        color_continuous_scale="RdYlGn",
        aspect="auto",
        labels=dict(color="NPV Equity"),
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Contorno — NPV Equity (línea de break-even)")

    figc = px.contour(
        heat_df,
        x="Exit",
        y="EBITDA",
        z="NPV",
        contours=dict(
            showlabels=True,
            coloring="none",
            start=0,
            end=0,
            size=1
        ),
    )
    figc.update_yaxes(autorange="reversed")
    st.plotly_chart(figc, use_container_width=True)

    # --- Heatmap IRR Equity ---
    heat_data_irr = []

    for e in ebitda_range:
        for ex in exit_range:
            if exit_method == "Cap rate sobre EBITDA":
                ev = e / (ex / 100.0)
            else:
                ev = e * ex

            exit_eq = ev - debt
            cf = e * (1 - tax_rate) - (debt * debt_rate)

            cashflows_temp = [-equity] + [cf] * (int(years) - 1) + [cf + exit_eq]
            irr_temp = irr(cashflows_temp, guess=max(disc_rate, 0.12) if disc_rate > 0 else 0.12)

            heat_data_irr.append({
                "EBITDA": e,
                "Exit": ex,
                "IRR": irr_temp * 100 if not math.isnan(irr_temp) else float("nan")
            })

    heat_df_irr = pd.DataFrame(heat_data_irr)
    pivot_irr = heat_df_irr.pivot(index="EBITDA", columns="Exit", values="IRR")

    fig2 = px.imshow(
        pivot_irr,
        color_continuous_scale="RdYlGn",
        aspect="auto",
        labels=dict(color="IRR Equity (%)"),
    )
    fig2.update_yaxes(autorange="reversed")
    st.plotly_chart(fig2, use_container_width=True)


# Diagnóstico opcional
with st.expander("Ver tabla del período (últimas 50 filas)"):
    st.dataframe(d.tail(50), use_container_width=True)