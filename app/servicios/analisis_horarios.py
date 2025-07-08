# app/servicios/analisis_horarios.py
import pandas as pd

def calcular_afluencia(path="data/tickets.csv"):
    tickets = pd.read_csv(path)
    est = pd.read_csv("data/estacionamientos.csv")  # ← Nuevo
    print("🔍 Columnas de estacionamientos:", est.columns)
    tickets["fechaValidacion"] = pd.to_datetime(tickets["fechaValidacion"], errors="coerce")
    tickets = tickets.dropna(subset=["fechaValidacion"])

    # Agregar columna con el nombre real del parqueadero
    tickets = tickets.merge(est[["id", "estacionamiento"]], left_on="parkingId", right_on="id", how="left")

    tickets["dia_semana"] = tickets["fechaValidacion"].dt.day_name()
    tickets["hora"] = tickets["fechaValidacion"].dt.hour

    dias_orden = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    tickets["dia_semana"] = pd.Categorical(tickets["dia_semana"], categories=dias_orden, ordered=True)

    df_afluencia = (
        tickets.groupby(["dia_semana", "hora", "estacionamiento"])  # no más 'nombre'
        .size()
        .reset_index(name="conteo")
        .sort_values(["dia_semana", "hora"])
    )


    return df_afluencia.to_dict(orient="records")

