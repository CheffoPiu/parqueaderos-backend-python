from servicios.cargar_datos import cargar_datos
from servicios.eda import eda

datos = cargar_datos()
eda(datos)

print("Clientes:", datos["clientes"].shape)
print("Facturas:", datos["facturas"].shape)
print("Tickets:", datos["tickets"].shape)
print("Comercios:", datos["comercios"].shape)
print("Estacionamientos:", datos["estacionamientos"].shape)
