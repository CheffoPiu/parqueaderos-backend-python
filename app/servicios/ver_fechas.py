import boto3
import pandas as pd
from datetime import datetime
from boto3.dynamodb.conditions import Attr
from boto3.session import Session

# --- Configurar perfil de AWS ---
session = Session(profile_name="spa")
dynamodb = session.resource("dynamodb", region_name="us-east-1")

# --- Configura tu tabla ---
TABLE_NAME = "TicketValidado-qnf3y7azksnd9jpkb8whgl0vmu-prod"
table = dynamodb.Table(TABLE_NAME)

# --- Fecha objetivo ---
fecha_objetivo = "2025-06-22"

print(f"🔍 Buscando todos los tickets con InitialDate en {fecha_objetivo}...")

# --- Escanear todas las páginas ---
tickets = []
last_evaluated_key = None

while True:
    scan_params = {
        "FilterExpression": Attr("InitialDate").contains(fecha_objetivo)
    }
    if last_evaluated_key:
        scan_params["ExclusiveStartKey"] = last_evaluated_key

    response = table.scan(**scan_params)
    items = response.get("Items", [])
    tickets.extend(items)

    last_evaluated_key = response.get("LastEvaluatedKey")
    if not last_evaluated_key:
        break

print(f"✅ Se encontraron {len(tickets)} tickets totales con InitialDate en {fecha_objetivo}.")

# --- Convertir a DataFrame y mostrar resumen por hora ---
if not tickets:
    print("❌ No se encontraron tickets.")
else:
    df = pd.DataFrame(tickets)
    df["InitialDate"] = pd.to_datetime(df["InitialDate"], errors="coerce").dt.tz_localize(None)
    df["hora"] = df["InitialDate"].dt.hour
    resumen = df.groupby("hora").size().reset_index(name="cantidad")

    print("\n📊 Resumen de tickets por hora:")
    print(resumen)
