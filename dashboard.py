import streamlit as st
from utils_data import load_snapshot, load_run_summary

st.set_page_config(
    page_title="Investment Lab",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🏦 Investment Lab — Home")

summary = load_run_summary()
if summary:
    st.caption(
        f"Último run (UTC): {summary.get('pipeline_run_utc', '—')} | "
        f"versión: {summary.get('pipeline_version', '—')}"
    )
else:
    st.caption("Sin run_summary disponible (igual puedes usar las páginas).")

st.divider()

st.markdown(
    """
**Navegación (menú izquierdo):**
- **Basics**: lectura rápida + tendencia 90 días.
- **Panel Rent + Hotel**: marco inmobiliario (renta + hotel).
- **Macro Comparativas**: relaciones macro (USD/DXY/TPM/UF).

Si el menú no aparece:
1) refresca la página  
2) Manage app → Reboot app  
"""
)

# Mini sanity check
with st.expander("Diagnóstico (snapshot)"):
    df = load_snapshot()
    st.write("Filas:", len(df))
    st.write("Rango:", df["fecha"].min().date(), "→", df["fecha"].max().date())
    st.dataframe(df.tail(10), use_container_width=True)