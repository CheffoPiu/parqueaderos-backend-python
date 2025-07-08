import pandas as pd

def cargar_datos():
    clientes = pd.read_csv("data/clientes.csv")
    comercios = pd.read_csv("data/comercios.csv")
    estacionamientos = pd.read_csv("data/estacionamientos.csv")
    facturas = pd.read_csv("data/facturas.csv")
    tickets = pd.read_csv("data/tickets.csv")

    return {
        "clientes": clientes,
        "comercios": comercios,
        "estacionamientos": estacionamientos,
        "facturas": facturas,
        "tickets": tickets
    }
