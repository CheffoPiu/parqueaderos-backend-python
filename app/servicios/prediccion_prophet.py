import pandas as pd
from prophet import Prophet

def transformar_tickets_dynamodb_a_prophet(items, estacionamiento_id=None):
    fechas_validas = []

    for item in items:
        fecha_raw = item.get("fechaValidacion")
        parking_id = item.get("parkingId")

        if not fecha_raw or (estacionamiento_id and parking_id != estacionamiento_id):
            continue

        # Convertir la fecha a datetime, si falla se vuelve NaT
        fecha = pd.to_datetime(fecha_raw, errors="coerce")

        if pd.notna(fecha):
            fechas_validas.append(fecha)

    if not fechas_validas:
        return pd.DataFrame()  # Devuelve DataFrame vacío si no hay fechas válidas

    df = pd.DataFrame(fechas_validas, columns=["ds"])
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    df = df.dropna(subset=["ds"])
    df["ds"] = df["ds"].dt.floor("H")
    df["ds"] = df["ds"].dt.tz_localize(None)

    df = df.groupby("ds").size().reset_index(name="y")
    return df

def preparar_datos_prophet(tickets_csv, estacionamiento_id=None):
    df = pd.read_csv(tickets_csv)
    df["fechaValidacion"] = pd.to_datetime(df["fechaValidacion"], errors="coerce")
    df = df.dropna(subset=["fechaValidacion"])

    if estacionamiento_id:
        df = df[df["parkingId"] == estacionamiento_id]

    df["hora"] = df["fechaValidacion"].dt.floor("H")
    df_agrupado = df.groupby("hora").size().reset_index(name="y")
    df_agrupado.rename(columns={"hora": "ds"}, inplace=True)

    return df_agrupado

def predecir_afluencia_prophet(df, dias_a_predecir=7):
    if df.empty:
        return {
            "prediccion": [],
            "error": {"mae": None, "mape": None, "n": 0}
        }

    # Detecta y filtra solo horas activas
    horas_activas = detectar_horas_activas(df)
    df = df[df['ds'].dt.hour.isin(horas_activas)].copy()

    modelo = Prophet()
    modelo.fit(df)

    # Crea el future dataframe SOLO con horas activas
    future = modelo.make_future_dataframe(periods=dias_a_predecir * 24, freq='H')
    future = future[future['ds'].dt.hour.isin(horas_activas)]  # <-- filtro clave

    forecast = modelo.predict(future)

    resultado = forecast[["ds", "yhat"]].copy()
    resultado["yhat"] = resultado["yhat"].clip(lower=0).round().astype(int)

    df_real = df[df["ds"] >= resultado["ds"].min()]
    error = calcular_error_prediccion(df_real, resultado) if not df_real.empty else {
        "mae": None, "mape": None, "n": 0
    }

    return {
        "prediccion": resultado.to_dict(orient="records"),
        "error": error
    }


def calcular_error_prediccion(real_df: pd.DataFrame, pred_df: pd.DataFrame):
    comparacion = real_df.merge(pred_df, on="ds", how="inner")

    comparacion["error_absoluto"] = (comparacion["y"] - comparacion["yhat"]).abs()
    comparacion["error_relativo"] = comparacion["error_absoluto"] / comparacion["y"].replace(0, 1)  # evita división por 0

    mae = comparacion["error_absoluto"].mean()
    mape = comparacion["error_relativo"].mean() * 100  # % error promedio

    return {
        "mae": round(mae, 2),
        "mape": round(mape, 2),
        "n": len(comparacion)
    }

def detectar_horas_activas(df, umbral=10, min_dias=0.2):
    """
    Detecta automáticamente las horas activas de un parqueadero.
    - umbral: cuántos usuarios/hora mínimo para considerarla activa.
    - min_dias: proporción mínima de días con actividad para considerarla.
    """
    if df.empty or 'ds' not in df.columns or 'y' not in df.columns:
        return []

    df = df.copy()
    df['hora'] = df['ds'].dt.hour
    df['fecha'] = df['ds'].dt.date

    actividad_por_hora = (
        df[df['y'] >= umbral]
        .groupby('hora')['fecha']
        .nunique()
        .sort_index()
    )
    total_dias = df['fecha'].nunique()
    horas_activas = actividad_por_hora[actividad_por_hora >= (min_dias * total_dias)].index.tolist()
    return horas_activas
