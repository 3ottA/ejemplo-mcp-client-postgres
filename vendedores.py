import os
import psycopg2
from random import randint, choice
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Conectar a la base de datos
load_dotenv()

conn = psycopg2.connect(
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    host=os.getenv("DB_HOST")
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
        id_campana = choice([None] + list(range(1, 26)))  # Algunas ventas sin campaña
        fecha_venta = random_date()
        monto = randint(100, 1000) + randint(0, 99) / 100  # Monto entre 100 y 1,000
        
        # Use column position instead of column name to avoid encoding issues
        query = "INSERT INTO ventas VALUES (DEFAULT, %s, %s, %s, %s)"
        cursor.execute(query, (id_vendedor, id_campana, fecha_venta, monto))

# ...existing code...   
# Confirmar cambios y cerrar conexión
conn.commit()
cursor.close()
conn.close()

print("Ventas insertadas correctamente.")