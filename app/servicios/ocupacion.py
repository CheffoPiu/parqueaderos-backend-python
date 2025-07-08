import pandas as pd
from datetime import datetime
from typing import List, Dict

# Capacidad estimada por parqueadero
CAPACIDAD_ESTIMADA = {
    "fdb74430-6fbd-4723-9ae0-4d0f1c887d6e": 37,
    "930c47ee-6ad2-4f9f-b91c-976920a1fb14": 43,
    "aa5b0034-5d85-4dec-8bcf-1009c7ee920f": 55,
    "c986e28c-e0af-4c05-8202-71e0972e0c31": 44,
    "0eb9fdd1-e7eb-4318-8b37-472a6fbbe22e": 26,
    "39e782ab-5a67-4945-8258-44aa7b96d460": 18,
}

# Día base para simular como si fuera hoy
DIA_BASE = "2024-04-15"

def obtener_ocupacion_actual_simulada() -> List[Dict]:
    now = datetime.now()

    tickets = pd.read_csv("data/tickets.csv", parse_dates=["InitialDate", "EndDate"])
    tickets = tickets.dropna(subset=["InitialDate", "EndDate", "parkingId"])

    dia_base_obj = datetime.strptime(DIA_BASE, "%Y-%m-%d").date()
    tickets = tickets[tickets["InitialDate"].dt.date == dia_base_obj]

   # ⚠️ Quitar zona horaria UTC si existe
    tickets["InitialDate"] = tickets["InitialDate"].dt.tz_localize(None)
    tickets["EndDate"] = tickets["EndDate"].dt.tz_localize(None)

    tickets_activos = tickets[(tickets["InitialDate"] <= now) & (tickets["EndDate"] > now)]

    resultados = []
    for parking_id, capacidad in CAPACIDAD_ESTIMADA.items():
        ocupados = len(tickets_activos[tickets_activos["parkingId"] == parking_id])
        porcentaje = round((ocupados / capacidad) * 100) if capacidad > 0 else 0

        if porcentaje < 50:
            color = "verde"
        elif porcentaje < 85:
            color = "amarillo"
        else:
            color = "rojo"

        resultados.append({
            "parkingId": parking_id,
            "ocupados": ocupados,
            "capacidad": capacidad,
            "porcentaje": porcentaje,
            "color": color
        })

    return resultados
