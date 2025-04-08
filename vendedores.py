import psycopg2
from random import randint, choice
from datetime import datetime, timedelta

# Conectar a la base de datos
conn = psycopg2.connect(
    dbname="ventas_db",
    user="postgres",  
    password="postgres",
    host="localhost"
)
cursor = conn.cursor()

# Distribución de ventas por vendedor
ventas_por_vendedor = [200, 180, 160, 140, 120, 100, 80, 60, 40, 20]

# Función para generar fecha aleatoria en 2023
def random_date():
    start = datetime(2025, 1, 1)
    delta = randint(0, 364)
    return start + timedelta(days=delta)

# Insertar ventas
for id_vendedor, num_ventas in enumerate(ventas_por_vendedor, 1):
    for _ in range(num_ventas):
        id_campaña = choice([None] + list(range(1, 26)))  # Algunas ventas sin campaña
        fecha_venta = random_date()
        monto = randint(100, 1000) + randint(0, 99) / 100  # Monto entre 100 y 1,000
        query = "INSERT INTO ventas (id_vendedor, id_campaña, fecha_venta, monto) VALUES (%s, %s, %s, %s)"
        cursor.execute(query, (id_vendedor, id_campaña, fecha_venta, monto))

# Confirmar cambios y cerrar conexión
conn.commit()
cursor.close()
conn.close()

print("Ventas insertadas correctamente.")