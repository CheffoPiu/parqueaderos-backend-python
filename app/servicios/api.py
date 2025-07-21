from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from app.servicios.analisis_horarios import calcular_afluencia
from fastapi.middleware.cors import CORSMiddleware
from app.servicios.preparar_dataset import preparar_dataset_facturas
from app.servicios.preparar_clientes import obtener_datos_clientes  # 👈 Importa esta nueva función
from app.servicios.ocupacion import obtener_ocupacion_actual
from app.servicios.parqueaderos import obtener_parqueaderos
from app.servicios.prediccion_prophet import preparar_datos_prophet, predecir_afluencia_prophet
from fastapi import APIRouter, Query, Body
from app.servicios.dynamodb_service import escanear_tickets
from app.servicios.prediccion_prophet import (
    transformar_tickets_dynamodb_a_prophet,
    predecir_afluencia_prophet
)
from typing import Optional
from openai import OpenAI
import os
from dotenv import load_dotenv


class ImagenRequest(BaseModel):
    base64_img: str
    prompt: str


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
app = FastAPI()
router = APIRouter()

# Habilitar CORS para que el frontend (Angular) acceda
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Puedes poner la URL exacta de tu frontend si prefieres
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar modelo
modelo = joblib.load("modelo_volvera.joblib")

# Esquema de entrada para predicción
class ClienteData(BaseModel):
    frecuencia: int
    recencia: int
    monto_total: float

@app.post("/predecir")
def predecir(data: ClienteData):
    entrada = np.array([[data.frecuencia, data.recencia, data.monto_total]])
    pred = modelo.predict(entrada)[0]
    prob = modelo.predict_proba(entrada)[0][1]
    return {
        "volvera": bool(pred),
        "probabilidad": round(float(prob), 4)
    }

# NUEVO: Endpoint para afluencia por hora y día
@app.get("/afluencia")
def afluencia():
    return calcular_afluencia()

@app.get("/clientes-probables")
def clientes_probables():
    df = preparar_dataset_facturas()
    clientes_df = obtener_datos_clientes()

    X = df[["frecuencia", "recencia", "monto_total"]]
    df["probabilidad"] = modelo.predict_proba(X)[:, 1]
    df["volvera"] = modelo.predict(X)

    # Enlaza con info de clientes
    df = df.merge(clientes_df, on="clienteId", how="left")

    columnas = [
        "clienteId", "nombre_completo", "email",
        "frecuencia", "recencia", "monto_total", "volvera", "probabilidad"
    ]

    df = df[columnas].sort_values(by="probabilidad", ascending=False).head(20)

    # Convertir valores no serializables a tipos nativos
    df = df.fillna("")  # Evita errores por NaN
    df = df.astype({
        "clienteId": str,
        "nombre_completo": str,
        "email": str,
        "frecuencia": int,
        "recencia": int,
        "monto_total": float,
        "volvera": int,
        "probabilidad": float
    })

    return df.to_dict(orient="records")

@app.get("/ocupacion-real")
def ocupacion_real():
    return obtener_ocupacion_actual()


@app.get("/clientes-segmentados")
def clientes_segmentados():
    df = preparar_dataset_facturas()

    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    # Datos de entrada para clustering
    X = df[["frecuencia", "recencia", "monto_total"]]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    df["grupo"] = kmeans.fit_predict(X_scaled)

    # Agregación por grupo
    resumen = df.groupby("grupo").agg(
        cantidad=("clienteId", "count"),
        frecuencia_promedio=("frecuencia", "mean"),
        recencia_promedio=("recencia", "mean"),
        monto_promedio=("monto_total", "mean")
    ).reset_index()

    # Etiquetas simples
    def etiquetar(row):
        if row["frecuencia_promedio"] > 5 and row["recencia_promedio"] < 30:
            return "Fieles"
        elif row["recencia_promedio"] > 100:
            return "Inactivos"
        else:
            return "Ocasionales"

    resumen["etiqueta"] = resumen.apply(etiquetar, axis=1)

    # Redondear valores
    resumen = resumen.round(2)

    return resumen.to_dict(orient="records")

@app.get("/parqueaderos")
def parqueaderos():
    return obtener_parqueaderos()

@app.get("/afluencia-predicha-prophet")
def afluencia_predicha_prophet(
    parqueadero_id: Optional[str] = Query(None),
    dias: int = 7
):
    items = escanear_tickets()
    df = transformar_tickets_dynamodb_a_prophet(items, estacionamiento_id=parqueadero_id)

    if df.empty:
        return {"error": "No hay datos suficientes para ese parqueadero"}

    prediccion = predecir_afluencia_prophet(df, dias_a_predecir=dias)
    return prediccion

@router.post("/interpretar-afluencia", tags=["IA"])
def interpretar_afluencia(datos: list = Body(...)):
    prompt = (
        "Analiza los siguientes datos de afluencia horaria en un parqueadero. "
        "Los datos están agrupados por hora y conteo de personas. "
        "Devuelve insights útiles para mejorar la gestión del parqueadero.\n\n"
        f"{datos}\n\n"
        "Devuelve un resumen claro y útil para un dashboard."
    )

    try:
        respuesta = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=500
        )
        return {"interpretacion": respuesta.choices[0].message.content}
    except Exception as e:
        return {"error": str(e)}
    
app.include_router(router)

@app.post("/analizar-imagen")
def analizar_imagen(data: ImagenRequest):
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": data.prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": data.base64_img
                            }
                        }
                    ]
                }
            ],
            max_tokens=800
        )
        return {"respuesta": response.choices[0].message.content}
    except Exception as e:
        return {"error": str(e)}

