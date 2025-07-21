import os
import boto3
import pandas as pd
from boto3.dynamodb.conditions import Attr
from boto3.session import Session
from dotenv import load_dotenv

load_dotenv()

PERFIL = os.getenv("AWS_PROFILE", "spa")
REGION = os.getenv("AWS_REGION", "us-east-1")
TABLE_NAME = os.getenv("PARQUEADEROS_TABLE", "Estacionamiento-qnf3y7azksnd9jpkb8whgl0vmu-prod")

session = Session(profile_name=PERFIL)
dynamodb = session.resource("dynamodb", region_name=REGION)
table = dynamodb.Table(TABLE_NAME)

def obtener_parqueaderos():
    response = table.scan()
    items = response.get("Items", [])
    if not items:
        return []

    df = pd.DataFrame(items)
    df = df.fillna("")

    parqueaderos = []
    for _, row in df.iterrows():
        parqueaderos.append({
            "parkingId": row.get("id", ""),
            "nombre": row.get("estacionamiento", ""),
            "lat": float(row["lat"]) if row.get("lat") not in [None, ""] else 0.0,
            "lng": float(row["lng"]) if row.get("lng") not in [None, ""] else 0.0,
            "capacidad": int(row.get("capacidad", 0)),
            "libres": 0,
            "color": "gray"
        })

    return parqueaderos
