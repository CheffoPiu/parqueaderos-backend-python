import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def eda(datos: dict):
    clientes = datos["clientes"]
    facturas = datos["facturas"]
    tickets = datos["tickets"]

    print("\n📊 Columnas en cada tabla:")
    print("- Clientes:", clientes.columns.tolist())
    print("- Facturas:", facturas.columns.tolist())
    print("- Tickets:", tickets.columns.tolist())

    print("\n🔢 Clientes únicos:", clientes["id"].nunique())
    print("🔁 Clientes en facturas:", facturas["clienteId"].nunique())
    print("🎟️ Tickets validados:", tickets.shape[0])

    # Histograma de valor facturado
    plt.figure(figsize=(8,4))
    sns.histplot(facturas["valorTotal"], bins=30, kde=True)
    plt.title("Distribución de valores facturados")
    plt.xlabel("Valor")
    plt.ylabel("Frecuencia")
    plt.tight_layout()
    plt.show()

    # ¿Frecuencia de visitas por cliente?
    visitas = facturas.groupby("clienteId").size().reset_index(name="frecuencia")
    print("\n📈 Promedio de visitas por cliente:", visitas["frecuencia"].mean())
    print("🧍 Clientes frecuentes (+5 visitas):", (visitas["frecuencia"] > 5).sum())
