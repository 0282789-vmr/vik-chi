import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from utils.gcs_loader import list_gcs_blobs, load_gcs_blob

st.title("Analítica Descriptiva desde GCS (Chicago)")

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
# Limpieza mínima (por si acaso)
# --------------------------------------------
if "trip_start_timestamp" in df.columns:
    df["trip_start_timestamp"] = pd.to_datetime(
        df["trip_start_timestamp"], errors="coerce"
    )

if "trip_seconds" in df.columns:
    df["trip_seconds"] = pd.to_numeric(df["trip_seconds"], errors="coerce")

# --------------------------------------------
# Histograma Altair: trip_seconds (p99)
# --------------------------------------------
st.subheader("Distribución de trip_seconds (p99)")

if "trip_seconds" in df.columns:
    # Calcular p99 ignorando NaN
    p99 = df["trip_seconds"].quantile(0.99)

    # Filtrar outliers
    df_plot = df[df["trip_seconds"] <= p99]

    # Solo por seguridad, evitar gráfico vacío
    if df_plot.empty:
        st.warning("No hay datos válidos para 'trip_seconds' después del filtrado.")
    else:
        chart = (
            alt.Chart(df_plot)
            .mark_bar()
            .encode(
                alt.X(
                    "trip_seconds:Q",
                    bin=True,
                    title="Duración (segundos)"
                ),
                alt.Y("count()", title="Frecuencia"),
            )
            .properties(height=350)
        )

        st.altair_chart(chart, use_container_width=True)
else:
    st.info("El archivo no contiene trip_seconds.")
