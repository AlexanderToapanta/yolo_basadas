import os
import sqlite3
from datetime import datetime

import pandas as pd


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_NAME = os.path.join(BASE_DIR, "tienda_logs.db")

ZONAS_MAPA_CALOR = (
    "superior_izquierda",
    "superior_centro",
    "superior_derecha",
    "centro_izquierda",
    "centro",
    "centro_derecha",
    "inferior_izquierda",
    "inferior_centro",
    "inferior_derecha",
)


def conectar():
    """Abre una conexión SQLite con espera breve ante escrituras simultáneas."""
    conn = sqlite3.connect(DB_NAME, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=10000")
    return conn


def init_db():
    """Crea las tablas requeridas sin eliminar los registros existentes."""
    with conectar() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fecha TEXT NOT NULL,
                hora TEXT NOT NULL,
                track_id INTEGER NOT NULL,
                evento TEXT NOT NULL CHECK (evento IN ('ENTRADA', 'SALIDA'))
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS heatmap_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fecha TEXT NOT NULL,
                hora TEXT NOT NULL,
                personas_detectadas INTEGER NOT NULL DEFAULT 0,
                aforo_estimado INTEGER NOT NULL DEFAULT 0,
                superior_izquierda REAL NOT NULL DEFAULT 0,
                superior_centro REAL NOT NULL DEFAULT 0,
                superior_derecha REAL NOT NULL DEFAULT 0,
                centro_izquierda REAL NOT NULL DEFAULT 0,
                centro REAL NOT NULL DEFAULT 0,
                centro_derecha REAL NOT NULL DEFAULT 0,
                inferior_izquierda REAL NOT NULL DEFAULT 0,
                inferior_centro REAL NOT NULL DEFAULT 0,
                inferior_derecha REAL NOT NULL DEFAULT 0,
                zona_mayor TEXT NOT NULL DEFAULT 'Sin actividad',
                concentracion_mayor REAL NOT NULL DEFAULT 0
            )
            """
        )

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_logs_fecha_hora ON logs(fecha, hora)"
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_heatmap_fecha_hora
            ON heatmap_snapshots(fecha, hora)
            """
        )


def insert_log(track_id, evento):
    """Inserta un cruce clasificado como ENTRADA o SALIDA."""
    evento = str(evento).upper().strip()
    if evento not in {"ENTRADA", "SALIDA"}:
        raise ValueError("El evento debe ser ENTRADA o SALIDA.")

    ahora = datetime.now()
    with conectar() as conn:
        conn.execute(
            """
            INSERT INTO logs (fecha, hora, track_id, evento)
            VALUES (?, ?, ?, ?)
            """,
            (
                ahora.strftime("%Y-%m-%d"),
                ahora.strftime("%H:%M:%S"),
                int(track_id),
                evento,
            ),
        )


def insert_heatmap_snapshot(
    zonas,
    personas_detectadas,
    aforo_estimado,
    zona_mayor,
    concentracion_mayor,
):
    """
    Guarda una muestra del mapa de calor.

    Los valores de `zonas` representan el porcentaje relativo de actividad
    visual de cada región y, cuando existe actividad, suman aproximadamente 100.
    No representan temperatura corporal ni una medición térmica.
    """
    faltantes = [zona for zona in ZONAS_MAPA_CALOR if zona not in zonas]
    if faltantes:
        raise ValueError(f"Faltan zonas del mapa de calor: {', '.join(faltantes)}")

    ahora = datetime.now()
    valores_zonas = [round(float(zonas[zona]), 4) for zona in ZONAS_MAPA_CALOR]

    with conectar() as conn:
        conn.execute(
            """
            INSERT INTO heatmap_snapshots (
                fecha,
                hora,
                personas_detectadas,
                aforo_estimado,
                superior_izquierda,
                superior_centro,
                superior_derecha,
                centro_izquierda,
                centro,
                centro_derecha,
                inferior_izquierda,
                inferior_centro,
                inferior_derecha,
                zona_mayor,
                concentracion_mayor
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ahora.strftime("%Y-%m-%d"),
                ahora.strftime("%H:%M:%S"),
                max(0, int(personas_detectadas)),
                int(aforo_estimado),
                *valores_zonas,
                str(zona_mayor),
                round(float(concentracion_mayor), 4),
            ),
        )


def get_filtered_logs(fecha, hora_inicio=None, hora_fin=None):
    """Devuelve los eventos de cruce para una fecha y rango horario."""
    query = "SELECT fecha, hora, track_id, evento FROM logs WHERE fecha = ?"
    params = [str(fecha)]

    if hora_inicio and hora_fin:
        query += " AND hora BETWEEN ? AND ?"
        params.extend([str(hora_inicio), str(hora_fin)])

    query += " ORDER BY hora"
    with conectar() as conn:
        return pd.read_sql_query(query, conn, params=params)


def get_heatmap_snapshots(fecha, hora_inicio=None, hora_fin=None):
    """Devuelve las muestras del mapa de calor para una fecha y rango horario."""
    query = "SELECT * FROM heatmap_snapshots WHERE fecha = ?"
    params = [str(fecha)]

    if hora_inicio and hora_fin:
        query += " AND hora BETWEEN ? AND ?"
        params.extend([str(hora_inicio), str(hora_fin)])

    query += " ORDER BY hora"
    with conectar() as conn:
        return pd.read_sql_query(query, conn, params=params)
