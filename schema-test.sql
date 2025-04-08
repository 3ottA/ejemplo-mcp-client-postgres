CREATE DATABASE ventas_db;

-- Conectar a la base de datos (en psql, usa: \c ventas_db)
-- Si estás en un script, asegúrate de conectar manualmente antes de continuar

-- Crear tabla Vendedores
CREATE TABLE Vendedores (
    id_vendedor SERIAL PRIMARY KEY,
    nombre VARCHAR(100),
    apellido VARCHAR(100),
    email VARCHAR(100)
);

-- Crear tabla Tipos_Campañas
CREATE TABLE Tipos_Campañas (
    id_tipo_campaña SERIAL PRIMARY KEY,
    nombre VARCHAR(100)
);

-- Crear tabla Campañas_Publicitarias
CREATE TABLE Campañas_Publicitarias (
    id_campaña SERIAL PRIMARY KEY,
    nombre VARCHAR(100),
    id_tipo_campaña INT,
    fecha_inicio DATE,
    fecha_fin DATE,
    CONSTRAINT fk_tipo_campaña FOREIGN KEY (id_tipo_campaña) REFERENCES Tipos_Campañas(id_tipo_campaña)
);

-- Crear tabla Ventas
CREATE TABLE Ventas (
    id_venta SERIAL PRIMARY KEY,
    id_vendedor INT,
    id_campaña INT,
    fecha_venta DATE,
    monto DECIMAL(10,2),
    CONSTRAINT fk_vendedor FOREIGN KEY (id_vendedor) REFERENCES Vendedores(id_vendedor),
    CONSTRAINT fk_campaña FOREIGN KEY (id_campaña) REFERENCES Campañas_Publicitarias(id_campaña)
);

-- Insertar datos en Vendedores (10 vendedores)
INSERT INTO Vendedores (nombre, apellido, email) VALUES
('Juan', 'Pérez', 'juan.perez@example.com'),
('María', 'González', 'maria.gonzalez@example.com'),
('Carlos', 'Rodríguez', 'carlos.rodriguez@example.com'),
('Ana', 'Martínez', 'ana.martinez@example.com'),
('Luis', 'Hernández', 'luis.hernandez@example.com'),
('Laura', 'López', 'laura.lopez@example.com'),
('Pedro', 'Sánchez', 'pedro.sanchez@example.com'),
('Sofía', 'Ramírez', 'sofia.ramirez@example.com'),
('Diego', 'Torres', 'diego.torres@example.com'),
('Elena', 'Vásquez', 'elena.vasquez@example.com');

-- Insertar datos en Tipos_Campañas (5 tipos)
INSERT INTO Tipos_Campañas (nombre) VALUES
('Email Marketing'),
('Facebook'),
('Instagram'),
('Google Ads'),
('Twitter');

-- Insertar datos en Campañas_Publicitarias (5 campañas por tipo, total 25 campañas)
INSERT INTO Campañas_Publicitarias (nombre, id_tipo_campaña, fecha_inicio, fecha_fin) VALUES
-- Email Marketing
('Campaña Email 1', 1, '2023-01-01', '2023-01-31'),
('Campaña Email 2', 1, '2023-02-01', '2023-02-28'),
('Campaña Email 3', 1, '2023-03-01', '2023-03-31'),
('Campaña Email 4', 1, '2023-04-01', '2023-04-30'),
('Campaña Email 5', 1, '2023-05-01', '2023-05-31'),
-- Facebook
('Campaña FB 1', 2, '2023-01-15', '2023-02-15'),
('Campaña FB 2', 2, '2023-03-01', '2023-03-31'),
('Campaña FB 3', 2, '2023-04-10', '2023-05-10'),
('Campaña FB 4', 2, '2023-06-01', '2023-06-30'),
('Campaña FB 5', 2, '2023-07-01', '2023-07-31'),
-- Instagram
('Campaña IG 1', 3, '2023-02-01', '2023-02-28'),
('Campaña IG 2', 3, '2023-03-15', '2023-04-15'),
('Campaña IG 3', 3, '2023-05-01', '2023-05-31'),
('Campaña IG 4', 3, '2023-06-15', '2023-07-15'),
('Campaña IG 5', 3, '2023-08-01', '2023-08-31'),
-- Google Ads
('Campaña GA 1', 4, '2023-01-01', '2023-01-31'),
('Campaña GA 2', 4, '2023-02-15', '2023-03-15'),
('Campaña GA 3', 4, '2023-04-01', '2023-04-30'),
('Campaña GA 4', 4, '2023-05-15', '2023-06-15'),
('Campaña GA 5', 4, '2023-07-01', '2023-07-31'),
-- Twitter
('Campaña TW 1', 5, '2023-03-01', '2023-03-31'),
('Campaña TW 2', 5, '2023-04-15', '2023-05-15'),
('Campaña TW 3', 5, '2023-06-01', '2023-06-30'),
('Campaña TW 4', 5, '2023-07-15', '2023-08-15'),
('Campaña TW 5', 5, '2023-09-01', '2023-09-30');