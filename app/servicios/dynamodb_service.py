import os
import boto3
from boto3.dynamodb.conditions import Attr
from boto3.session import Session
from dotenv import load_dotenv

load_dotenv()

PERFIL = os.getenv("AWS_PROFILE", "spa")
REGION = os.getenv("AWS_REGION", "us-east-1")
TABLE_NAME = os.getenv("TICKETS_TABLE", "TicketValidado-qnf3y7azksnd9jpkb8whgl0vmu-prod")

session = Session(profile_name=PERFIL)
dynamodb = session.resource("dynamodb", region_name=REGION)
table = dynamodb.Table(TABLE_NAME)

CACHE_TICKETS = []

def escanear_tickets():
    global CACHE_TICKETS
    if CACHE_TICKETS:
        return CACHE_TICKETS

    print("🔄 Escaneando tickets desde DynamoDB...")
    response = table.scan()
    items = response.get("Items", [])
    CACHE_TICKETS = items
    return items
