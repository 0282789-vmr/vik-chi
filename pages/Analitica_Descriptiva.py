import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from utils.gcs_loader import list_gcs_blobs, load_gcs_blob
import pydeck as pdk

st.title("Analítica Descriptiva desde GCS (Chicago Taxi)")

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

        #st.caption(f"Filas totales: {len(df_plot)} · Filas graficadas (muestra): {n}")

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
    # Asegurar tipo numérico 
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

        #st.caption(
            #f"[trip_miles] Filas totales: {len(df_plot_miles)} · "
            #f"Filas graficadas (muestra): {n_miles}"
        #)

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
# --------------------------------------------
# Distribución de payment_type (porcentajes)
# --------------------------------------------
st.subheader("Distribución de tipos de pago")

if "payment_type" in df.columns:
    # 1) Limpiar: a string, quitar espacios, NaN -> 'UNKNOWN'
    s = df["payment_type"].astype("string").str.strip()
    s = s.fillna("UNKNOWN")
    s = s.replace({"": "UNKNOWN"})

    # 2) Normalizar algunas variantes comunes 
    #    Primero pasamos todo a minúsculas para mapear:
    s_lower = s.str.lower()

    mapping = {
        "cash": "Cash",
        "efectivo": "Cash",          # por si hubiera
        "credit card": "Credit Card",
        "creditcard": "Credit Card",
        "tarjeta": "Credit Card",    # por si hubiera
        "prcard": "Prepaid Card",
        "precard": "Prepaid Card",
        "mobile": "Mobile",
        "unknown": "UNKNOWN",
    }

    # Aplicar mapping manteniendo el valor original cuando no esté en el diccionario
    s_norm = s_lower.map(mapping).fillna(s)

    # 3) Agrupar: una fila por categoría
    df_pay = (
        s_norm.value_counts(dropna=False)
        .reset_index()
    )
    df_pay.columns = ["payment_type", "count"]

    # 4) Calcular porcentaje
    total = df_pay["count"].sum()
    df_pay["percent"] = df_pay["count"] / total * 100

    # 5) Gráfico de barras por porcentaje
    chart_pay = (
        alt.Chart(df_pay)
        .mark_bar()
        .encode(
            x=alt.X(
                "payment_type:N",
                title="Tipo de pago",
                sort="-y",  # ordenar de mayor a menor porcentaje
            ),
            y=alt.Y(
                "percent:Q",
                title="Porcentaje de viajes",
                axis=alt.Axis(format=".1f")
            ),
            tooltip=[
                alt.Tooltip("payment_type:N", title="Tipo de pago"),
                alt.Tooltip("percent:Q", title="Porcentaje", format=".2f"),
                alt.Tooltip("count:Q", title="Número de viajes"),
            ],
        )
        .properties(
            height=350,
            title="Distribución de tipos de pago (porcentaje)"
        )
    )

    st.altair_chart(chart_pay, use_container_width=True)

    # 
    st.caption("Tipos de pago")
    st.dataframe(df_pay.sort_values("percent", ascending=False), use_container_width=True)

else:
    st.info("El archivo no contiene la columna 'payment_type'.")

# ---------------------------------------------------------
# Mapa PyDeck: rutas alrededor de la media de trip_total
# ---------------------------------------------------------
st.subheader("Rutas alrededor de la media de trip_total")

cols_needed = [
    "trip_total",
    "pickup_latitude", "pickup_longitude",
    "dropoff_latitude", "dropoff_longitude",
]

# Verificamos que existan las columnas necesarias
if set(cols_needed).issubset(df.columns):

    # Aseguramos tipo numérico
    for col in cols_needed:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Filtrar a filas con coordenadas completas
    df_geo = df.dropna(
        subset=[
            "pickup_latitude", "pickup_longitude",
            "dropoff_latitude", "dropoff_longitude",
        ]
    ).copy()

    if df_geo.empty:
        st.warning("No hay viajes con coordenadas completas para el mapa.")
    else:
        # Media de trip_total usando solo viajes con coordenadas válidas
        media = df_geo["trip_total"].mean()

        # 50 más cercanos por debajo de la media
        df_below = (
            df_geo[df_geo["trip_total"] < media]
            .sort_values("trip_total", ascending=False)
            .head(50)
        )

        # 50 más cercanos por arriba de la media
        df_above = (
            df_geo[df_geo["trip_total"] >= media]
            .sort_values("trip_total", ascending=True)
            .head(50)
        )

        df_map = pd.concat([df_below, df_above], ignore_index=True)

        st.caption(f"Viajes seleccionados para el mapa: {len(df_map)}")

        if df_map.empty:
            st.warning("No se encontraron suficientes viajes para graficar.")
        else:
            # Colores: rojo = arriba de la media, verde = abajo
            df_map["color"] = df_map["trip_total"].apply(
                lambda x: [255, 0, 0] if x > media else [0, 200, 0]
            )

            # Capa de arcos pickup -> dropoff
            arc_layer = pdk.Layer(
                "ArcLayer",
                data=df_map,
                get_source_position=["pickup_longitude", "pickup_latitude"],
                get_target_position=["dropoff_longitude", "dropoff_latitude"],
                get_source_color="color",
                get_target_color="color",
                auto_highlight=True,
                width_scale=2,
                get_width=2,
            )

            # Vista centrada en Chicago
            view_state = pdk.ViewState(
                latitude=41.8781,
                longitude=-87.6298,
                zoom=10,
                pitch=0,
                bearing=0,
            )

            deck = pdk.Deck(
                layers=[arc_layer],
                initial_view_state=view_state,
                tooltip={
                    "text": "trip_total: {trip_total}\n"
                            "pickup: [{pickup_latitude}, {pickup_longitude}]\n"
                            "dropoff: [{dropoff_latitude}, {dropoff_longitude}]"
                },
            )

            st.pydeck_chart(deck)
else:
    st.info(
        "Faltan columnas para el mapa: "
        "'trip_total', 'pickup_latitude', 'pickup_longitude', "
        "'dropoff_latitude', 'dropoff_longitude'."
    )
