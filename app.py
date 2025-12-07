import streamlit as st

st.set_page_config(
    page_title="Dashboard en la Nube – Cloud Run",
    layout="wide"
)

st.title("Dashboard – Analítica Descriptiva en la Nube (Cloud Run + GCS)")

st.markdown("""
Bienvenido al dashboard en la nube desplegado con **Cloud Run**.

Este sistema permite:

### Analizar datasets grandes alojados en Google Cloud Storage
- Cargar archivos CSV desde un bucket.
- Navegar archivo por archivo.
- Visualizar histogramas, patrones horarios y matrices de correlación usando **Altair**.

---

Usa el menú lateral para acceder a la sección:

**1_Analitica_Descriptiva_GCS**

---
""")
