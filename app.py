import streamlit as st

st.set_page_config(
    page_title="Chicago Taxi Dashboard",
    layout="wide"
)

st.title("Chicago Taxi Dashboard")

st.markdown("""
Chicago Taxi Dashboard.

Esta aplicación permite:

### Analizar datasets grandes alojados en Google Cloud Storage
- Cargar archivos CSV desde un bucket.
- Navegar archivo por archivo.
- Visualizar histogramas, tablas, mapas **Altair**.
- Modelo de regresión lineal incremental con River

---

Usa el menú lateral para acceder a la sección:

- Analitica_Descriptiva
- Modelo_Incremental_Chicago

---
""")
