import pandas as pd
import requests
import time

df = pd.read_csv("data/estacionamientos.csv")
df["lat"] = None
df["lng"] = None

for i, row in df.iterrows():
    direccion = f"{row['direccion']}, {row['ciudad']}, Ecuador"
    print(f"🔍 Geolocalizando: {direccion}")
    
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": direccion,
        "format": "json",
        "limit": 1
    }
    headers = {
        "User-Agent": "GeoParqueaderosApp"
    }

    try:
        response = requests.get(url, params=params, headers=headers)
        data = response.json()
        if data:
            lat = data[0]["lat"]
            lng = data[0]["lon"]
            df.at[i, "lat"] = lat
            df.at[i, "lng"] = lng
            print(f"✅ {row['estacionamiento']} → {lat}, {lng}")
        else:
            print(f"⚠️ No se encontró: {direccion}")
    except Exception as e:
        print(f"❌ Error con {direccion}: {e}")

    time.sleep(1)  # para no saturar la API

# Guardar el resultado
df.to_csv("data/estacionamientos_geolocalizados.csv", index=False)
print("✅ Guardado: estacionamientos_geolocalizados.csv")
