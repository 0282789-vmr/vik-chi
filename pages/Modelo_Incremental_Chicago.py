import io
import numpy as np
import pandas as pd
import streamlit as st

from google.cloud import storage
from river import linear_model, preprocessing, metrics


# =========================================================
# Configuración de la página
# =========================================================
st.title("Modelo Incremental (River) – Viajes de Taxi en Chicago")


# =========================================================
# 1. Parámetros de conexión y entrenamiento
# =========================================================
st.sidebar.header("Configuración")

bucket_name = st.sidebar.text_input("Bucket de GCS", "0282789_bucket")
prefix = st.sidebar.text_input("Prefijo", "chi_trips/")

limite_muestras = st.sidebar.number_input(
    "Máximo de muestras POR archivo",
    min_value=1_000,
    max_value=100_000,
    value=10_000,
    step=1_000,
)

chunksize = st.sidebar.number_input(
    "Tamaño de chunk (filas por lectura)",
    min_value=100,
    max_value=5_000,
    value=1_000,
    step=100,
)

reiniciar_modelo = st.sidebar.checkbox(
    "Reiniciar modelo desde cero en esta ejecución",
    value=True,
)


# =========================================================
# 2. Funciones auxiliares (versión Chicago)
# =========================================================
def _parse_time_fields_chi(row):
    """Devuelve (timestamp, hour, day, month, weekday) a partir de trip_start_timestamp."""
    if "trip_start_timestamp" in row and pd.notna(row["trip_start_timestamp"]):
        dt = pd.to_datetime(row["trip_start_timestamp"], errors="coerce", utc=False)
        if pd.notna(dt):
            return dt, int(dt.hour), int(dt.day), int(dt.month), int(dt.weekday())
    # fallback
    return None, 0, 1, 1, 0


def _extract_x_chi(row):
    """
    Construye el vector de características X para Chicago.

    - trip_miles, trip_seconds
    - pickup/dropoff_community_area
    - hour, day, month, weekday
    """
    miles = row.get("trip_miles", 0)
    miles = float(pd.to_numeric(miles, errors="coerce")) if pd.notna(miles) else 0.0

    secs = row.get("trip_seconds", 0)
    secs = float(pd.to_numeric(secs, errors="coerce")) if pd.notna(secs) else 0.0

    dt, hour, day, month, weekday = _parse_time_fields_chi(row)

    pca = row.get("pickup_community_area", 0)
    pca = float(pd.to_numeric(pca, errors="coerce")) if pd.notna(pca) else 0.0

    dca = row.get("dropoff_community_area", 0)
    dca = float(pd.to_numeric(dca, errors="coerce")) if pd.notna(dca) else 0.0

    return {
        "trip_miles": miles,
        "log_miles": float(np.log1p(max(miles, 0.0))),
        "trip_seconds": secs,
        "log_seconds": float(np.log1p(max(secs, 0.0))),
        "pickup_area": pca,
        "dropoff_area": dca,
        "hour": float(hour),
        "day": float(day),
        "month": float(month),
        "weekday": float(weekday),
    }


def _valid_target_chi(v):
    """Limpia y valida trip_total como objetivo."""
    y = pd.to_numeric(v, errors="coerce")
    if pd.isna(y):
        return None
    y = float(y)
    if not np.isfinite(y):
        return None
    return y


# =========================================================
# 3. Entrenamiento incremental desde GCS (límite POR archivo)
# =========================================================
def train_incremental_from_bucket_chicago(
    bucket_name: str,
    prefix: str,
    limite_por_archivo: int,      # máximo de muestras por archivo
    chunksize: int,
    model=None,
    metric=None,
    logger=None,
):
    """
    Entrena un modelo de regresión lineal incremental (River) usando
    los CSV de Chicago en GCS.

    - limite_por_archivo: máximo de muestras **por archivo**.
      Se recorren todos los archivos del prefijo.
    """

    if logger is None:
        logger = print

    # Inicializar modelo/ métrica si no vienen de fuera
    if model is None or metric is None:
        model = preprocessing.StandardScaler() | linear_model.LinearRegression()
        metric = metrics.R2()
    history = []

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # Listar blobs y quedarnos solo con los .csv
    all_blobs = list(bucket.list_blobs(prefix=prefix))
    blobs = [b for b in all_blobs if b.name.endswith(".csv")]

    expected_cols = {
        "trip_miles",
        "trip_seconds",
        "trip_total",
        "trip_start_timestamp",
        "pickup_community_area",
        "dropoff_community_area",
    }

    logger(f"📦 Archivos encontrados en {bucket_name}/{prefix}: {len(blobs)}")

    for i, blob in enumerate(blobs, start=1):
        logger(f"\nArchivo {i}/{len(blobs)} — {blob.name}")

        # contador POR ARCHIVO (como en tu código de NYC)
        count = 0

        content = blob.download_as_bytes()
        buffer = io.BytesIO(content)

        try:
            for chunk in pd.read_csv(buffer, chunksize=chunksize, low_memory=False):

                if count >= limite_por_archivo:
                    break

                if not expected_cols.issubset(chunk.columns):
                    logger("   → Saltando (faltan columnas esperadas)")
                    break

                for col in [
                    "trip_miles",
                    "trip_seconds",
                    "trip_total",
                    "pickup_community_area",
                    "dropoff_community_area",
                ]:
                    chunk[col] = pd.to_numeric(chunk[col], errors="coerce")

                chunk = chunk.replace([np.inf, -np.inf], np.nan)
                chunk = chunk.dropna(
                    subset=["trip_miles", "trip_seconds", "trip_total"]
                )

                chunk = chunk[
                    chunk["trip_total"].between(2, 200)
                    & chunk["trip_miles"].between(0.1, 50)
                    & chunk["trip_seconds"].between(60, 7200)
                ]

                if chunk.empty:
                    continue

                chunk = chunk.sample(frac=1.0, random_state=42)

                for _, row in chunk.iterrows():
                    if count >= limite_por_archivo:
                        break

                    y = _valid_target_chi(row.get("trip_total"))
                    if y is None:
                        continue

                    x = _extract_x_chi(row)

                    y_pred = model.predict_one(x)
                    model.learn_one(x, y)
                    metric.update(y, y_pred)

                    count += 1

                    if count % 500 == 0:
                        logger(f"   → {count} muestras (R²={metric.get():.3f})")

        except Exception as e:
            logger(f"Error procesando {blob.name}: {e}")
            continue

        logger(f"R² tras este archivo: {metric.get():.3f}")
        history.append(metric.get())

    logger(
        f"\n✅ Entrenamiento finalizado. Se usaron hasta {limite_por_archivo} "
        f"muestras por archivo. R² final = {metric.get():.3f}"
    )
    return model, history, metric


# =========================================================
# 4. Interfaz en Streamlit (botón, logs, gráfico)
# =========================================================
if st.button("Entrenar modelo incremental"):
    log_area = st.empty()

    # Inicializamos lista de logs en session_state
    if "log_lines_chi" not in st.session_state:
        st.session_state["log_lines_chi"] = []

    def st_logger(msg: str):
        """Acumula logs y los muestra en Streamlit."""
        st.session_state["log_lines_chi"].append(str(msg))
        ultimas = st.session_state["log_lines_chi"][-40:]
        log_area.text("\n".join(ultimas))

    with st.spinner("Entrenando modelo incremental con River..."):
        # Reusar o reiniciar modelo
        if (
            "river_model_chi" not in st.session_state
            or "river_metric_chi" not in st.session_state
            or reiniciar_modelo
        ):
            current_model = None
            current_metric = None
            st.session_state["log_lines_chi"] = []
        else:
            current_model = st.session_state["river_model_chi"]
            current_metric = st.session_state["river_metric_chi"]

        model, history, metric = train_incremental_from_bucket_chicago(
            bucket_name=bucket_name,
            prefix=prefix,
            limite_por_archivo=limite_muestras,  # 👈 aquí usas 10k por archivo
            chunksize=chunksize,
            model=current_model,
            metric=current_metric,
            logger=st_logger,
        )

        st.session_state["river_model_chi"] = model
        st.session_state["river_metric_chi"] = metric

    st.success(f"Entrenamiento completo. R² final = {metric.get():.3f}")

    if history:
        st.subheader("Evolución de R² por archivo procesado")
        hist_df = pd.DataFrame({"R2": history})
        st.line_chart(hist_df)
else:
    st.info(
        "Configura los parámetros en la barra lateral y pulsa "
        "**Entrenar modelo incremental**."
    )
