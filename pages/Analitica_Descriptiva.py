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
# Limpieza mínima
# --------------------------------------------
if "trip_start_timestamp" in df.columns:
    df["trip_start_timestamp"] = pd.to_datetime(
        df["trip_start_timestamp"], errors="coerce"
    )

if "trip_seconds" in df.columns:
    df["trip_seconds"] = pd.to_numeric(df["trip_seconds"], errors="coerce")

# --------------------------------------------
# Distribución de trip_seconds (p99)
# --------------------------------------------
st.subheader("Distribución de trip_seconds (p99)")

if "trip_seconds" in df.columns:
    series = df["trip_seconds"]

    # Calcular p99 ignorando NaN
    p99 = series.quantile(0.99)

    # Filtrar NaN y outliers
    df_plot = df[(series.notna()) & (series <= p99)].copy()

    # Limitar a máximo 5000 filas para Altair
    n = min(5000, len(df_plot))
    if n == 0:
        st.warning("No hay datos válidos para 'trip_seconds' después del filtrado.")
    else:
        df_plot_sample = df_plot.sample(n=n, random_state=42)

        st.caption(f"Filas totales: {len(df_plot)} · Filas graficadas (muestra): {n}")

        chart = (
            alt.Chart(df_plot_sample)
            .mark_bar()
            .encode(
                alt.X(
                    "trip_seconds:Q",
                    bin=alt.Bin(maxbins=50),
                    title="Duración (segundos)"
                ),
                alt.Y("count()", title="Frecuencia"),
            )
            .properties(height=350)
        )

        st.altair_chart(chart, use_container_width=True)
else:
    st.info("El archivo no contiene 'trip_seconds'.")

# --------------------------------------------
# Distribución de trip_miles (p99)
# --------------------------------------------
st.subheader("Distribución de trip_miles (p99)")

if "trip_miles" in df.columns:
    # Asegurar tipo numérico (por si acaso)
    df["trip_miles"] = pd.to_numeric(df["trip_miles"], errors="coerce")
    series_miles = df["trip_miles"]

    # Calcular p99 ignorando NaN
    p99_miles = series_miles.quantile(0.99)

    # Filtrar NaN y outliers
    df_plot_miles = df[(series_miles.notna()) & (series_miles <= p99_miles)].copy()

    # Muestrear máximo 5000 filas para Altair
    n_miles = min(5000, len(df_plot_miles))
    if n_miles == 0:
        st.warning("No hay datos válidos para 'trip_miles' después del filtrado.")
    else:
        df_plot_miles_sample = df_plot_miles.sample(n=n_miles, random_state=42)

        st.caption(
            f"[trip_miles] Filas totales: {len(df_plot_miles)} · "
            f"Filas graficadas (muestra): {n_miles}"
        )

        chart_miles = (
            alt.Chart(df_plot_miles_sample)
            .mark_bar()
            .encode(
                alt.X(
                    "trip_miles:Q",
                    bin=alt.Bin(maxbins=50),
                    title="Distancia (millas)"
                ),
                alt.Y("count()", title="Frecuencia"),
            )
            .properties(height=350)
        )

        st.altair_chart(chart_miles, use_container_width=True)
else:
    st.info("El archivo no contiene 'trip_miles'.")
