import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")


def eda(datos):
    clientes = datos["clientes"]
    facturas = datos["facturas"]
    tickets = datos["tickets"]
    comercios = datos["comercios"]
    estacionamientos = datos["estacionamientos"]

    print("\n📊 Columnas en cada tabla:")
    for nombre, df in datos.items():
        print(f"- {nombre.capitalize()}: {df.columns.tolist()}")

    print("\n🔢 Clientes únicos:", clientes["id"].nunique())

    if "clienteId" not in facturas.columns:
        print("\n🔁 La columna clienteId no está disponible directamente en facturas.")
    else:
        print("\n🔁 Clientes en facturas:", facturas["clienteId"].nunique())

    print("🎟️ Tickets validados:", len(tickets))

    # Histograma de montos si existe 'valorTicket' o 'valor'
    if "valorTicket" in tickets.columns:
        plt.figure(figsize=(8, 4))
        sns.histplot(tickets["valorTicket"], bins=30, kde=True, color="skyblue")
        plt.title("Distribuci\u00f3n de valores de tickets")
        plt.xlabel("Valor Ticket")
        plt.ylabel("Frecuencia")
        plt.tight_layout()
        plt.show()
    elif "valor" in facturas.columns:
        plt.figure(figsize=(8, 4))
        sns.histplot(facturas["valor"], bins=30, kde=True, color="orange")
        plt.title("Distribuci\u00f3n de valores de facturas")
        plt.xlabel("Valor Factura")
        plt.ylabel("Frecuencia")
        plt.tight_layout()
        plt.show()
    else:
        print("⚠️ No se encontró la columna 'valor' para graficar distribución.")

    # Análisis de frecuencia por cliente
    if "clienteId" in facturas.columns:
        visitas = facturas.groupby("clienteId").size().reset_index(name="frecuencia")
        print("\n📈 Promedio de visitas por cliente:", visitas["frecuencia"].mean())
        plt.figure(figsize=(8, 4))
        sns.histplot(visitas["frecuencia"], bins=30, kde=False, color="green")
        plt.title("Frecuencia de visitas por cliente")
        plt.xlabel("Número de visitas")
        plt.ylabel("Cantidad de clientes")
        plt.tight_layout()
        plt.show()
    else:
        print("⚠️ No se puede agrupar por 'clienteId' porque no existe directamente en 'facturas'.")
