import streamlit as st
import pandas as pd
from utils_data import load_snapshot

st.set_page_config(page_title="Macro Comparativas", layout="wide")

st.title("📊 Macro Comparativas")

df = load_snapshot()

min_date = df["fecha"].min().date()
max_date = df["fecha"].max().date()

# Selector rango
start_default = min_date
end_default = max_date

start, end = st.date_input(
    "Rango",
    value=(start_default, end_default),
    min_value=min_date,
    max_value=max_date,
)

mask = (df["fecha"].dt.date >= start) & (df["fecha"].dt.date <= end)
d = df.loc[mask].copy()

if len(d) < 5:
    st.warning("Rango muy corto. Elige un periodo más largo.")
    st.stop()

st.subheader("USD/CLP vs DXY (Base 100 al inicio del rango)")
base = d.iloc[0]
d["usd_base100"] = (d["usd_clp"] / base["usd_clp"]) * 100
d["dxy_base100"] = (d["dxy"] / base["dxy"]) * 100

st.line_chart(d.set_index("fecha")[["usd_base100", "dxy_base100"]])

st.subheader("TPM vs USD/CLP (niveles)")
st.line_chart(d.set_index("fecha")[["usd_clp", "tpm"]])

with st.expander("Tabla (últimas 200 filas del rango)"):
    st.dataframe(d.tail(200), use_container_width=True)