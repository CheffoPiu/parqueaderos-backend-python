from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from servicios.cargar_datos import cargar_datos
from servicios.eda import eda

app = FastAPI()

# CORS para permitir Angular (localhost:4200)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar y procesar datos
datos = cargar_datos()
eda(datos)

# Tus endpoints irán aquí...
