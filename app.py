import streamlit as st

st.set_page_config(
    page_title="Dashboard en la Nube – Cloud Run",
    layout="wide"
)

st.title("Dashboard – Analítica Descriptiva")

st.markdown("""
Bienvenido al dashboard en la nube desplegado con **Cloud Run**.

Este sistema permite:

### Analizar datasets grandes alojados en Google Cloud Storage
- Cargar archivos CSV desde un bucket.
- Navegar archivo por archivo.
- Visualizar histogramas, patrones horarios y matrices de correlación usando **Altair**.

---

Usa el menú lateral para acceder a la sección:

**Analitica_Descriptiva**
**Modelo_Incremental_Chicago**

---
""")
