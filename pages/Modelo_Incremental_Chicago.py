import io
import numpy as np
import pandas as pd
import streamlit as st

from google.cloud import storage
from river import linear_model, preprocessing, metrics


# =========================================================
# Página: Modelo incremental (River) - Chicago
# =========================================================
st.title("Modelo Incremental (River) – Viajes de Taxi en Chicago")


# =========================================================
# 1. Parámetros de conexión y entrenamiento
# =========================================================
st.sidebar.header("Configuración")

bucket_name = st.sidebar.text_input("Bucket de GCS", "0282789_bucket")
prefix = st.sidebar.text_input("Prefijo", "chi_trips/")

limite_muestras = st.sidebar.number_input(
    "Límite total de muestras para entrenar",
    min_value=1000,
    max_value=100_000,
    value=20_000,
    step=1000,
)

chunksize = st.sidebar.number_input(
    "Tamaño de chunk (filas por lectura)",
    min_value=100,
    max_value=5000,
    value=1000,
    step=100,
)

reiniciar_modelo = st.sidebar.checkbox(
    "Reiniciar modelo desde cero en esta ejecución", value=True
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
    # Distancia
    miles = row.get("trip_miles", 0)
    miles = float(pd.to_numeric(miles, errors="coerce")) if pd.notna(miles) else 0.0

    # Duración
    secs = row.get("trip_seconds", 0)
    secs = float(pd.to_numeric(secs, errors="coerce")) if pd.notna(secs) else 0.0

    # Timestamp → variables temporales
    dt, hour, day, month, weekday = _parse_time_fields_chi(row)

    # Community areas
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
# 3. Entrenamiento incremental desde GCS (Chicago)
# =========================================================
def train_incremental_from_bucket_chicago(
    bucket_name: str,
    prefix: str,
    limite: int,
    chunksize: int,
    model=None,
    metric=None,
    logger=None,
):
    """
    Entrena un modelo de regresión lineal incremental (River) usando
    los CSV de Chicago en GCS.

    - bucket_name: nombre del bucket de GCS
    - prefix: prefijo de los archivos (ej. 'chi_trips/')
    - limite: número máximo total de muestras a usar
    - chunksize: tamaño de cada chunk leído por pandas
    - model, metric: si se pasan, continúa entrenando; si no, los crea
    - logger: función para imprimir mensajes en Streamlit
    """
    if logger is None:
        logger = print

    # Si no se pasa un modelo, se inicializa desde cero
    if model is None or metric is None:
        model = preprocessing.StandardScaler() | linear_model.LinearRegression()
        metric = metrics.R2()
        history = []
    else:
        history = []

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix))

    expected_cols = {
        "trip_miles",
        "trip_seconds",
        "trip_total",
        "trip_start_timestamp",
        "pickup_community_area",
        "dropoff_community_area",
    }

    logger(f"📦 Archivos encontrados en {bucket_name}/{prefix}: {len(blobs)}")

    total_count = 0

    for i, blob in enumerate(blobs, start=1):
        if total_count >= limite:
            break

        logger(f"\nArchivo {i}/{len(blobs)} — {blob.name}")

        content = blob.download_as_bytes()
        buffer = io.BytesIO(content)

        try:
            for chunk in pd.read_csv(buffer, chunksize=chunksize, low_memory=False):

                if total_count >= limite:
                    break

                # Validar columnas mínimas
                if not expected_cols.issubset(chunk.columns):
                    logger("   → Saltando (faltan columnas esperadas)")
                    break

                # Conversión numérica
                for col in [
                    "trip_miles",
                    "trip_seconds",
                    "trip_total",
                    "pickup_community_area",
                    "dropoff_community_area",
                ]:
                    chunk[col] = pd.to_numeric(chunk[col], errors="coerce")

                # Limpiar NaN e infinitos
                chunk = chunk.replace([np.inf, -np.inf], np.nan)
                chunk = chunk.dropna(
                    subset=["trip_miles", "trip_seconds", "trip_total"]
                )

                # Filtros razonables
                chunk = chunk[
                    chunk["trip_total"].between(2, 200)
                    & chunk["trip_miles"].between(0.1, 50)
                    & chunk["trip_seconds"].between(60, 7200)
                ]

                if chunk.empty:
                    continue

                # Shuffle interno
                chunk = chunk.sample(frac=1.0, random_state=42)

                # Entrenamiento online
                for _, row in chunk.iterrows():
                    if total_count >= limite:
                        break

                    y = _valid_target_chi(row.get("trip_total"))
                    if y is None:
                        continue

                    x = _extract_x_chi(row)

                    y_pred = model.predict_one(x)
                    model.learn_one(x, y)
                    metric.update(y, y_pred)

                    total_count += 1

                    if total_count % 500 == 0:
                        logger(f"   → {total_count} muestras (R²={metric.get():.3f})")

        except Exception as e:
            logger(f"Error procesando {blob.name}: {e}")
            continue

        logger(f"R² tras este archivo: {metric.get():.3f}")
        history.append(metric.get())

    logger(
        f"\n✅ Entrenamiento finalizado con {total_count} muestras. "
        f"R² final = {metric.get():.3f}"
    )
    return model, history, metric


# =========================================================
# 4. Interfaz en Streamlit
# =========================================================
if st.button("Entrenar modelo incremental"):
    log_area = st.empty()
    log_lines = []

    def st_logger(msg: str):
        """Acumula logs y los muestra en Streamlit."""
        nonlocal log_lines
        log_lines.append(str(msg))
        # Mostrar solo las últimas 40 líneas para no saturar
        log_area.text("\n".join(log_lines[-40:]))

    with st.spinner("Entrenando modelo incremental con River..."):
        # Si queremos reusar el modelo entre ejecuciones,
        # podemos guardarlo en session_state
        if (
            "river_model_chi" not in st.session_state
            or "river_metric_chi" not in st.session_state
            or reiniciar_modelo
        ):
            current_model = None
            current_metric = None
        else:
            current_model = st.session_state["river_model_chi"]
            current_metric = st.session_state["river_metric_chi"]

        model, history, metric = train_incremental_from_bucket_chicago(
            bucket_name=bucket_name,
            prefix=prefix,
            limite=limite_muestras,
            chunksize=chunksize,
            model=current_model,
            metric=current_metric,
            logger=st_logger,
        )

        st.session_state["river_model_chi"] = model
        st.session_state["river_metric_chi"] = metric

    st.success(f"Entrenamiento completo. R² final = {metric.get():.3f}")

    # Mostrar historial de R² si hay datos
    if history:
        st.subheader("Evolución de R² por archivo procesado")
        hist_df = pd.DataFrame({"R2": history})
        st.line_chart(hist_df)
else:
    st.info("Configura los parámetros en la barra lateral y pulsa **Entrenar modelo incremental**.")
