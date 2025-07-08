import pandas as pd
import json
from datetime import datetime

def preparar_dataset_facturas(path="data/facturas.csv"):
    facturas = pd.read_csv(path)

    # Extraer clienteId desde campo JSON tipo DynamoDB
    def extraer_cliente_id(dc):
        try:
            if isinstance(dc, dict):
                return dc.get("id", {}).get("S")
            elif isinstance(dc, str):
                cliente = json.loads(dc.replace("'", '"'))
                return cliente.get("id", {}).get("S")
        except Exception:
            return None

    facturas["clienteId"] = facturas["datosCliente"].apply(extraer_cliente_id)
    facturas = facturas.dropna(subset=["clienteId"])
    facturas = facturas[facturas["clienteId"].apply(lambda x: isinstance(x, str))]

    # Convertir fecha
    facturas["fecha"] = pd.to_datetime(facturas["fecha"], errors="coerce").dt.tz_localize(None)
    facturas = facturas.dropna(subset=["fecha"])

    # Extraer valor desde detallesFactura
    def extraer_valor(df):
        try:
            if isinstance(df, str):
                items = json.loads(df.replace("'", '"'))
            else:
                items = df
            return float(items[0]["M"]["total"]["N"])
        except Exception:
            return 0

    facturas["valor"] = facturas["detallesFactura"].apply(extraer_valor)

    # Ordenar
    facturas = facturas.sort_values(by=["clienteId", "fecha"])
    facturas["siguiente_visita"] = facturas.groupby("clienteId")["fecha"].shift(-1)
    facturas["dias_entre_visitas"] = (facturas["siguiente_visita"] - facturas["fecha"]).dt.days

    # Etiqueta: volverá en 30 días
    facturas["volvera"] = facturas["dias_entre_visitas"].apply(
        lambda x: 1 if pd.notna(x) and x <= 30 else 0
    )

    # Agregación por cliente
    df_modelo = facturas.groupby("clienteId").agg(
        frecuencia=("id", "count"),
        recencia=("fecha", lambda x: (datetime.now() - x.max()).days),
        monto_total=("valor", "sum"),
        volvera=("volvera", "max")
    ).reset_index()

    # DEBUG opcional
    print("\n📊 Registros finales en dataset de entrenamiento:", len(df_modelo))
    print(df_modelo.head())

    return df_modelo
