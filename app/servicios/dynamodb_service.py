import os
import pandas as pd
import boto3
from boto3.session import Session
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

PERFIL = os.getenv("AWS_PROFILE", "spa")
REGION = os.getenv("AWS_REGION", "us-east-1")
TABLE_NAME = os.getenv("TICKETS_TABLE", "TicketValidado-qnf3y7azksnd9jpkb8whgl0vmu-prod")

session = Session(profile_name=PERFIL)
dynamodb = session.resource("dynamodb", region_name=REGION)
table = dynamodb.Table(TABLE_NAME)

CACHE_TICKETS = []

def guardar_tickets_csv(items, archivo_csv="tickets_cache.csv"):
    df = pd.DataFrame(items)
    df.to_csv(archivo_csv, index=False)
    print(f"✅ Tickets guardados en {archivo_csv}")

def cargar_tickets_desde_csv(archivo_csv="tickets_cache.csv"):
    df = pd.read_csv(archivo_csv)
    items = df.to_dict(orient="records")
    print(f"✅ Tickets cargados desde {archivo_csv}: {len(items)}")
    return items

def escanear_tickets(force_reload=False, archivo_csv="tickets_cache.csv"):
    global CACHE_TICKETS
    if not force_reload and os.path.exists(archivo_csv):
        print("🔵 Cargando tickets desde CSV local...")
        CACHE_TICKETS = cargar_tickets_desde_csv(archivo_csv)
        return CACHE_TICKETS

    print("🔄 Escaneando tickets desde DynamoDB...")
    items = []
    response = table.scan()
    items.extend(response.get("Items", []))

    while 'LastEvaluatedKey' in response:
        response = table.scan(ExclusiveStartKey=response['LastEvaluatedKey'])
        items.extend(response.get("Items", []))
        print(f"  - Total acumulado: {len(items)}")

    CACHE_TICKETS = items
    guardar_tickets_csv(items, archivo_csv)
    print(f"✅ Total tickets escaneados: {len(items)}")
    return items