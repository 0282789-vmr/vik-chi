import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from utils.gcs_loader import list_gcs_blobs, load_gcs_blob
import folium
from streamlit_folium import st_folium

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

# --------------------------------------------------------------------
# Mapa: 50 viajes debajo y 50 arriba de la media de trip_total
# --------------------------------------------------------------------
st.subheader("Rutas alrededor de la media de trip_total (mapa)")

cols_needed = {
    "trip_total",
    "pickup_latitude", "pickup_longitude",
    "dropoff_latitude", "dropoff_longitude",
}

if cols_needed.issubset(df.columns):

    # Asegurar tipos numéricos
    for col in cols_needed:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    media = df["trip_total"].mean()

    # 50 más cercanos por debajo
    df_below = (
        df[df["trip_total"] < media]
        .sort_values("trip_total", ascending=False)
        .head(50)
    )

    # 50 más cercanos por arriba
    df_above = (
        df[df["trip_total"] >= media]
        .sort_values("trip_total", ascending=True)
        .head(50)
    )

    df_map = pd.concat([df_below, df_above], ignore_index=True)

    # Eliminar filas sin coordenadas
    df_map = df_map.dropna(
        subset=[
            "pickup_latitude", "pickup_longitude",
            "dropoff_latitude", "dropoff_longitude",
        ]
    ).reset_index(drop=True)

    st.caption(f"Viajes seleccionados tras limpieza: {len(df_map)}")

    if df_map.empty:
        st.warning("No hay viajes con coordenadas completas para trazar líneas.")
    else:
        # Crear mapa centrado en Chicago
        m = folium.Map(location=[41.8781, -87.6298], zoom_start=11, tiles="CartoDB Positron")

        # Dibujar líneas pickup -> dropoff
        for _, row in df_map.iterrows():
            pickup = [row["pickup_latitude"], row["pickup_longitude"]]
            dropoff = [row["dropoff_latitude"], row["dropoff_longitude"]]

            color = "green" if row["trip_total"] > media else "red"

            folium.PolyLine(
                locations=[pickup, dropoff],
                color=color,
                weight=3,
                opacity=0.6,
            ).add_to(m)

        # Renderizar mapa en streamlit
        st_folium(m, width=800, height=500)

else:
    st.info(
        "Faltan columnas para el mapa: "
        "'trip_total', 'pickup_latitude', 'pickup_longitude', "
        "'dropoff_latitude', 'dropoff_longitude'."
    )


