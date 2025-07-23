import pandas as pd

def calcular_afluencia(path="tickets_cache.csv", fecha_inicio=None, fecha_fin=None, parqueadero_id=None):
    tickets = pd.read_csv(path)
    est = pd.read_csv("data/estacionamientos.csv")
    # Convierte a datetime y elimina zona horaria (si la tuviera)
    tickets["fechaValidacion"] = pd.to_datetime(tickets["fechaValidacion"], errors="coerce")
    tickets["fechaValidacion"] = tickets["fechaValidacion"].dt.tz_localize(None)


    # Filtro de fechas
    if fecha_inicio:
        tickets = tickets[tickets["fechaValidacion"] >= pd.to_datetime(fecha_inicio)]
    if fecha_fin:
        tickets = tickets[tickets["fechaValidacion"] <= pd.to_datetime(fecha_fin)]

    # Filtro de parqueadero por id
    if parqueadero_id:
        tickets = tickets[tickets["parkingId"] == parqueadero_id]

    # Merge para obtener el nombre real solo si quieres mostrarlo
    tickets = tickets.merge(est[["id", "estacionamiento"]], left_on="parkingId", right_on="id", how="left")

    tickets["dia_semana"] = tickets["fechaValidacion"].dt.day_name()
    tickets["hora"] = tickets["fechaValidacion"].dt.hour

    dias_orden = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    tickets["dia_semana"] = pd.Categorical(tickets["dia_semana"], categories=dias_orden, ordered=True)

    df_afluencia = (
        tickets.groupby(["dia_semana", "hora", "estacionamiento"])
        .size()
        .reset_index(name="conteo")
        .sort_values(["dia_semana", "hora"])
    )   

    return df_afluencia.to_dict(orient="records")


def preparar_datos_prophet(tickets_csv, estacionamiento_id=None):
    df = pd.read_csv(tickets_csv)
    df["fechaValidacion"] = pd.to_datetime(df["fechaValidacion"], errors="coerce")
    df = df.dropna(subset=["fechaValidacion"])

    if estacionamiento_id:
        df = df[df["parkingId"] == estacionamiento_id]

    df["hora"] = df["fechaValidacion"].dt.floor("H")

    df_agrupado = df.groupby("hora").size().reset_index(name="y")
    df_agrupado.rename(columns={"hora": "ds"}, inplace=True)

    # ⚠️ Quitar zona horaria
    df_agrupado["ds"] = df_agrupado["ds"].dt.tz_localize(None)

    return df_agrupado



