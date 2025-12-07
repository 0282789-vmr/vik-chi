import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from utils.gcs_loader import list_gcs_blobs, load_gcs_blob


# Desactivar límite de 5000 filas de Altair
alt.data_transformers.disable_max_rows()

st.title("Analítica Descriptiva desde GCS")

# --------------------------------------------
# Parámetros de conexión
# --------------------------------------------
bucket = st.text_input("Bucket de GCS:", "0282789_bucket")
prefix = st.text_input("Prefijo:", "chi_trips/")

# --------------------------------------------
# Listar blobs en el bucket
# --------------------------------------------
blobs = list_gcs_blobs(bucket, prefix)

if not blobs:
    st.error("No se encontraron archivos .csv en ese prefijo.")
    st.stop()

# Estado persistente del índice
if "blob_index" not in st.session_state:
    st.session_state.blob_index = 0

# Botón para avanzar
if st.button("Procesar siguiente archivo"):
    if st.session_state.blob_index + 1 < len(blobs):
        st.session_state.blob_index += 1
    else:
        st.warning("No hay más archivos disponibles.")

actual_blob = blobs[st.session_state.blob_index]
st.success(f"Analizando: **{actual_blob}**")

# --------------------------------------------
# Cargar datos
# --------------------------------------------
df = load_gcs_blob(bucket, actual_blob)

st.subheader("Vista previa")
st.dataframe(df.head())

# --------------------------------------------
# Limpieza mínima
# --------------------------------------------
if "trip_seconds" in df.columns:
    df["trip_seconds"] = pd.to_numeric(df["trip_seconds"], errors="coerce")

# --------------------------------------------
# Distribución de trip_seconds (p99)
# --------------------------------------------
st.subheader("Distribución de trip_seconds (p99)")

if "trip_seconds" in df.columns:
    # Percentil 99
    p99 = df["trip_seconds"].quantile(0.99)

    # Filtrar outliers por arriba del p99
    df_plot = df[df["trip_seconds"] <= p99]

    # Histograma con Altair
    chart = (
        alt.Chart(df_plot)
        .mark_bar()
        .encode(
            alt.X("trip_seconds:Q",
                  bin=alt.Bin(maxbins=50),
                  title="Duración (segundos)"),
            alt.Y("count()", title="Frecuencia"),
        )
        .properties(height=350)
    )

    st.altair_chart(chart, use_container_width=True)
else:
    st.info("El archivo no contiene la columna 'trip_seconds'.")
