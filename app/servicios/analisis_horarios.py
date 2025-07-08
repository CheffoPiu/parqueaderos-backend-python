# app/servicios/analisis_horarios.py
import pandas as pd

def calcular_afluencia(path="data/tickets.csv"):
    tickets = pd.read_csv(path)
    tickets["fechaValidacion"] = pd.to_datetime(tickets["fechaValidacion"], errors="coerce")
    tickets = tickets.dropna(subset=["fechaValidacion"])

    tickets["dia_semana"] = tickets["fechaValidacion"].dt.day_name()
    tickets["hora"] = tickets["fechaValidacion"].dt.hour

    dias_orden = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    tickets["dia_semana"] = pd.Categorical(tickets["dia_semana"], categories=dias_orden, ordered=True)

    df_afluencia = (
        tickets.groupby(["dia_semana", "hora"])
        .size()
        .reset_index(name="conteo")
        .sort_values(["dia_semana", "hora"])
    )

    # Convertimos a lista de diccionarios para API
    return df_afluencia.to_dict(orient="records")
