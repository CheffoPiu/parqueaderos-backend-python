from servicios.cargar_datos import cargar_datos
from servicios.eda import eda

datos = cargar_datos()

print("Clientes:", datos["clientes"].shape)
print("Facturas:", datos["facturas"].shape)
print("Tickets:", datos["tickets"].shape)

eda(datos)