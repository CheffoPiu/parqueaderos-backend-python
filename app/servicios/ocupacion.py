import os
import json
import pandas as pd
from datetime import datetime
from typing import List, Dict
from dotenv import load_dotenv
import boto3
from boto3.dynamodb.conditions import Attr
from boto3.session import Session

load_dotenv()

# --- Configuración ---
CAPACIDAD_ESTIMADA = json.loads(os.getenv("CAPACIDAD_ESTIMADA", "{}"))
DIA_BASE = os.getenv("DIA_BASE", "2025-06-22")
fecha_objetivo = datetime.strptime(DIA_BASE, "%Y-%m-%d").date()
CACHE_TICKETS: List[Dict] = []

PERFIL = os.getenv("AWS_PROFILE", "spa")
REGION = os.getenv("AWS_REGION", "us-east-1")
TABLE_NAME = os.getenv("TICKETS_TABLE", "TicketValidado-qnf3y7azksnd9jpkb8whgl0vmu-prod")

# --- Sesión AWS ---
session = Session(profile_name=PERFIL)
dynamodb = session.resource("dynamodb", region_name=REGION)
table = dynamodb.Table(TABLE_NAME)

# 🔁 Escanea toda la tabla con filtro por fecha (paginado)
def escanear_completo_por_fecha(fecha: str) -> List[Dict]:
    items = []
    kwargs = {
        "FilterExpression": Attr("InitialDate").contains(fecha)
    }

    while True:
        response = table.scan(**kwargs)
        items.extend(response.get("Items", []))

        if "LastEvaluatedKey" not in response:
            break

        kwargs["ExclusiveStartKey"] = response["LastEvaluatedKey"]

    return items

# 🔄 Simulación de ocupación en tiempo real
def obtener_ocupacion_actual() -> List[Dict]:
    global CACHE_TICKETS

    now = datetime.now()

    # Solo carga desde Dynamo una vez
    if not CACHE_TICKETS:
        print(f"🔎 Cargando desde Dynamo los tickets del día {DIA_BASE}")
        CACHE_TICKETS = escanear_completo_por_fecha(DIA_BASE)

    if not CACHE_TICKETS:
        print("❌ No se encontraron tickets para esa fecha.")
        return []

    df = pd.DataFrame(CACHE_TICKETS)
    df["InitialDate"] = pd.to_datetime(df["InitialDate"], errors="coerce").dt.tz_localize(None)
    df["EndDate"] = pd.to_datetime(df["EndDate"], errors="coerce").dt.tz_localize(None)
    df = df.dropna(subset=["InitialDate", "EndDate", "parkingId"])

    df["hora_inicio"] = df["InitialDate"].dt.hour
    resumen_horas = df.groupby("hora_inicio").size().reset_index(name="vehiculos")
    print("📊 Vehículos por hora:")
    print(resumen_horas)

    # Simula el tiempo actual dentro del día objetivo
    now_simulado = now.replace(
        year=fecha_objetivo.year,
        month=fecha_objetivo.month,
        day=fecha_objetivo.day
    )

    tickets_activos = df[(df["InitialDate"] <= now_simulado) & (df["EndDate"] > now_simulado)]

    resultados = []
    for parking_id, capacidad in CAPACIDAD_ESTIMADA.items():
        ocupados = len(tickets_activos[tickets_activos["parkingId"] == parking_id])
        porcentaje = round((ocupados / capacidad) * 100) if capacidad > 0 else 0

        color = (
            "verde" if porcentaje < 50 else
            "amarillo" if porcentaje < 85 else
            "rojo"
        )

        resultados.append({
            "parkingId": parking_id,
            "ocupados": ocupados,
            "capacidad": capacidad,
            "porcentaje": porcentaje,
            "color": color
        })

    return resultados

