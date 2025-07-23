import pandas as pd
from prophet import Prophet

# Transforma los tickets a un DataFrame para Prophet, filtrando por estacionamiento si se indica
def transformar_tickets_dynamodb_a_prophet(items, estacionamiento_id=None):
    fechas_validas = []

    for item in items:
        fecha_raw = item.get("fechaValidacion")
        parking_id = item.get("parkingId")

        # Filtra si falta la fecha o no coincide el estacionamiento (si se usa filtro)
        if not fecha_raw or (estacionamiento_id and parking_id != estacionamiento_id):
            continue

        # Convertir la fecha a datetime, si falla se vuelve NaT
        fecha = pd.to_datetime(fecha_raw, errors="coerce")

        if pd.notna(fecha):
            fechas_validas.append(fecha)

    if not fechas_validas:
        return pd.DataFrame()  # Devuelve DataFrame vacío si no hay fechas válidas

    # Arma DataFrame para Prophet
    df = pd.DataFrame(fechas_validas, columns=["ds"])
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    df = df.dropna(subset=["ds"])
    df["ds"] = df["ds"].dt.floor("H")
    df["ds"] = df["ds"].dt.tz_localize(None)

    df = df.groupby("ds").size().reset_index(name="y")
    df["y"] = df["y"].clip(upper=df["y"].quantile(0.99)) 
    return df

# Prepara los datos para Prophet a partir de un CSV de tickets (opcionalmente filtrando por estacionamiento)
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

# Predice la afluencia con Prophet y calcula el error sobre el rango con datos reales
def predecir_afluencia_prophet(df, dias_a_predecir=7):
    if df.empty:
        return {
            "prediccion": [],
            "error": {"mae": None, "mape": None, "n": 0}
        }

    # --- Detección de horas activas por día de la semana
    horas_activas_por_dia = detectar_horas_activas_por_dia_semana(df)
    df['hora'] = df['ds'].dt.hour
    df['dia_semana'] = df['ds'].dt.dayofweek
    df = df[df.apply(lambda x: x['hora'] in horas_activas_por_dia.get(x['dia_semana'], []), axis=1)].copy()
    if df.empty:
        return {
            "prediccion": [],
            "error": {"mae": None, "mape": None, "n": 0}
        }

    # === Aquí debes entrenar el modelo Prophet ===
    modelo = Prophet(
        seasonality_mode="additive",
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False
    )
    modelo.add_seasonality(name='weekly', period=7, fourier_order=10)
    modelo.add_seasonality(name='daily', period=1, fourier_order=8)
    modelo.fit(df)

    # --- FUTURE filtrado SOLO en combinaciones válidas
    future = modelo.make_future_dataframe(periods=dias_a_predecir * 24, freq='H')
    future['hora'] = future['ds'].dt.hour
    future['dia_semana'] = future['ds'].dt.dayofweek
    future = future[future.apply(lambda x: x['hora'] in horas_activas_por_dia.get(x['dia_semana'], []), axis=1)]

    forecast = modelo.predict(future)
    resultado = forecast[["ds", "yhat"]].copy()
    resultado["yhat"] = resultado["yhat"].clip(lower=0).round().astype(int)

    # Solo evalúa el error donde hay datos reales (periodo solapado)
    df_real = df[df["ds"] >= resultado["ds"].min()]
    error = calcular_error_prediccion(df_real, resultado) if not df_real.empty else {
        "mae": None, "mape": None, "n": 0
    }

    return {
        "prediccion": resultado.to_dict(orient="records"),
        "error": error
    }


# Calcula el error MAE y MAPE entre los datos reales y la predicción
def calcular_error_prediccion(real_df: pd.DataFrame, pred_df: pd.DataFrame, min_y=10):
    comparacion = real_df.merge(pred_df, on="ds", how="inner")

    comparacion["error_absoluto"] = (comparacion["y"] - comparacion["yhat"]).abs()
    comparacion["error_relativo"] = comparacion["error_absoluto"] / comparacion["y"].replace(0, 1)  # sigue evitando división por 0

    mae = comparacion["error_absoluto"].mean()
    # Solo cuenta MAPE donde el valor real >= min_y
    comparacion_filtrada = comparacion[comparacion["y"] >= min_y]
    if len(comparacion_filtrada) > 0:
        mape = comparacion_filtrada["error_relativo"].mean() * 100
    else:
        mape = None

    return {
        "mae": round(mae, 2),
        "mape": round(mape, 2) if mape is not None else None,
        "n": len(comparacion)
    }

# Detecta horas activas por día de la semana (devuelve un dict: {día: [horas activas]})
def detectar_horas_activas_por_dia_semana(df, umbral=1, min_dias=0.1):
    """
    Devuelve un dict: {dia_semana: [horas_activas, ...]}
    """
    if df.empty or 'ds' not in df.columns or 'y' not in df.columns:
        return {}
    df = df.copy()
    df['hora'] = df['ds'].dt.hour
    df['dia_semana'] = df['ds'].dt.dayofweek  # 0=Lunes, ..., 6=Domingo
    df['fecha'] = df['ds'].dt.date

    horas_por_dia = {}
    for dia in range(7):
        subset = df[df['dia_semana'] == dia]
        actividad = (
            subset[subset['y'] >= umbral]
            .groupby('hora')['fecha']
            .nunique()
            .sort_index()
        )
        total_dias = subset['fecha'].nunique()
        horas_activas = actividad[actividad >= (min_dias * total_dias)].index.tolist()
        horas_por_dia[dia] = horas_activas
    return horas_por_dia


# Detecta horas activas globales (no por día de la semana, solo por hora)
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
