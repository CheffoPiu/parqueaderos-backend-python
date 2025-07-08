import pandas as pd
import os
import json
from datetime import datetime


def normalizar_json_string(s):
    try:
        if isinstance(s, str):
            return json.loads(s.replace("'", '"'))
        return s
    except Exception:
        return None


def cargar_datos(base_path="data"):
    print("\n📂 Cargando y limpiando datos...")

    # Leer CSVs
    clientes = pd.read_csv(os.path.join(base_path, "clientes.csv"))
    facturas = pd.read_csv(os.path.join(base_path, "facturas.csv"))
    tickets = pd.read_csv(os.path.join(base_path, "tickets.csv"))
    comercios = pd.read_csv(os.path.join(base_path, "comercios.csv"))
    estacionamientos = pd.read_csv(os.path.join(base_path, "estacionamientos.csv"))

    # Unificar nombres de columnas
    clientes = clientes.rename(columns={"tipoID": "tipoId"})

    # Convertir fechas
    for df in [clientes, facturas, tickets]:
        for col in ["fecha", "createdAt", "InitialDate", "EndDate", "fechaValidacion"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

    # Limpiar JSONs como datosCliente
    if "datosCliente" in facturas.columns:
        facturas["datosCliente"] = facturas["datosCliente"].apply(normalizar_json_string)
        facturas["clienteId"] = facturas["datosCliente"].apply(lambda d: d.get("id") if isinstance(d, dict) else None)
        facturas["clienteId"] = facturas["clienteId"].astype(str)

    # Eliminar duplicados y vacíos claves
    facturas = facturas.dropna(subset=["clienteId", "fecha"])
    tickets = tickets.dropna(subset=["ticketId", "createdAt"])

    print("✅ Clientes:", clientes.shape)
    print("✅ Facturas:", facturas.shape)
    print("✅ Tickets:", tickets.shape)
    print("✅ Comercios:", comercios.shape)
    print("✅ Estacionamientos:", estacionamientos.shape)

    return {
        "clientes": clientes,
        "facturas": facturas,
        "tickets": tickets,
        "comercios": comercios,
        "estacionamientos": estacionamientos,
    }

if __name__ == "__main__":
    datos = cargar_datos()
    for key, df in datos.items():
        print(f"{key}: {df.shape}")
