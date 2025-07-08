import pandas as pd

def obtener_datos_clientes(path="data/clientes.csv"):
    df = pd.read_csv(path, dtype={"id": str})
    df = df.rename(columns={"id": "clienteId"})
    df["nombre_completo"] = df["nombre"].fillna("") + " " + df["apellido"].fillna("")
    return df[["clienteId", "nombre_completo", "email"]]
