import json
import os
import shutil
import sqlite3
import subprocess
import sys
from datetime import datetime, time

import pandas as pd
import streamlit as st

import database as db


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AGENTE_JS = os.path.join(BASE_DIR, "agente_resumen.js")
ENV_PATH = os.path.join(BASE_DIR, ".env")

ZONAS_COLUMNAS = {
    "superior_izquierda": "Superior izquierda",
    "superior_centro": "Superior centro",
    "superior_derecha": "Superior derecha",
    "centro_izquierda": "Centro izquierda",
    "centro": "Centro",
    "centro_derecha": "Centro derecha",
    "inferior_izquierda": "Inferior izquierda",
    "inferior_centro": "Inferior centro",
    "inferior_derecha": "Inferior derecha",
}


def cargar_archivo_env(ruta):
    """Carga variables sencillas desde .env sin dependencias externas."""
    if not os.path.exists(ruta):
        return

    with open(ruta, "r", encoding="utf-8") as archivo:
        for linea in archivo:
            linea = linea.strip()
            if not linea or linea.startswith("#") or "=" not in linea:
                continue

            clave, valor = linea.split("=", 1)
            clave = clave.strip()
            valor = valor.strip().strip('"').strip("'")
            os.environ.setdefault(clave, valor)


def encontrar_node():
    """Localiza Node.js mediante PATH o en rutas comunes de Windows."""
    encontrado = shutil.which("node")
    if encontrado:
        return encontrado

    candidatos = [
        r"C:\Program Files\nodejs\node.exe",
        r"C:\Program Files (x86)\nodejs\node.exe",
    ]
    for candidato in candidatos:
        if os.path.isfile(candidato):
            return candidato

    raise RuntimeError(
        "No se encontró Node.js. Agrégalo al PATH o verifica que exista "
        r"C:\Program Files\nodejs\node.exe"
    )


def generar_resumen_con_agente(metricas):
    """Ejecuta el agente JavaScript y devuelve el resumen generado por Groq."""
    cargar_archivo_env(ENV_PATH)
    node = encontrar_node()

    if not os.path.exists(AGENTE_JS):
        raise FileNotFoundError(
            f"No se encontró el agente JavaScript en: {AGENTE_JS}"
        )

    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError(
            "Falta GROQ_API_KEY. Crea un archivo .env junto a dashboard.py."
        )

    if not os.getenv("GROQ_MODEL"):
        raise RuntimeError(
            "Falta GROQ_MODEL. Agrégalo al archivo .env junto a dashboard.py."
        )

    proceso = subprocess.run(
        [node, AGENTE_JS],
        input=json.dumps(metricas, ensure_ascii=False),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
        cwd=BASE_DIR,
        env=os.environ.copy(),
        check=False,
    )

    salida = proceso.stdout.strip()
    if not salida:
        detalle = proceso.stderr.strip() or "El agente JavaScript no devolvió respuesta."
        raise RuntimeError(detalle)

    try:
        resultado = json.loads(salida)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"La respuesta del agente no es JSON válido: {salida[:300]}"
        ) from exc

    if not resultado.get("ok"):
        raise RuntimeError(resultado.get("error", "No se pudo generar el resumen."))

    return resultado["resumen"]


def cargar_datos():
    """Carga los eventos y las muestras espaciales desde SQLite."""
    db.init_db()
    with sqlite3.connect(db.DB_NAME) as conn:
        logs = pd.read_sql_query("SELECT * FROM logs ORDER BY fecha, hora", conn)
        heatmap = pd.read_sql_query(
            "SELECT * FROM heatmap_snapshots ORDER BY fecha, hora", conn
        )
    return logs, heatmap


def preparar_fechas(df, columna_fecha="fecha"):
    if df.empty:
        return df
    copia = df.copy()
    copia[columna_fecha] = pd.to_datetime(
        copia[columna_fecha], errors="coerce"
    ).dt.date
    return copia.dropna(subset=[columna_fecha])


def filtrar_por_hora(df, hora_inicio, hora_fin):
    if df.empty:
        return df

    copia = df.copy()
    copia["hora_dt"] = pd.to_datetime(
        copia["hora"], format="%H:%M:%S", errors="coerce"
    ).dt.time
    copia = copia.dropna(subset=["hora_dt"])
    return copia[
        (copia["hora_dt"] >= hora_inicio) & (copia["hora_dt"] <= hora_fin)
    ]


def analizar_mapa_calor(df_heatmap):
    """Resume las muestras espaciales del periodo seleccionado."""
    if df_heatmap.empty:
        return {
            "disponible": False,
            "zona_mayor": "No determinada",
            "concentracion_mayor": 0.0,
            "distribucion": {nombre: 0.0 for nombre in ZONAS_COLUMNAS.values()},
            "muestras": 0,
            "promedio_personas": 0.0,
            "aforo_promedio": 0.0,
        }

    promedios = df_heatmap[list(ZONAS_COLUMNAS.keys())].mean()
    distribucion = {
        ZONAS_COLUMNAS[columna]: round(float(promedios[columna]), 2)
        for columna in ZONAS_COLUMNAS
    }

    columna_mayor = max(ZONAS_COLUMNAS, key=lambda col: float(promedios[col]))
    concentracion_mayor = round(float(promedios[columna_mayor]), 2)

    disponible = bool(concentracion_mayor > 0)
    return {
        "disponible": disponible,
        "zona_mayor": (
            ZONAS_COLUMNAS[columna_mayor] if disponible else "Sin actividad"
        ),
        "concentracion_mayor": concentracion_mayor if disponible else 0.0,
        "distribucion": distribucion,
        "muestras": int(len(df_heatmap)),
        "promedio_personas": round(
            float(df_heatmap["personas_detectadas"].mean()), 2
        ),
        "aforo_promedio": round(float(df_heatmap["aforo_estimado"].mean()), 2),
    }


def mostrar_cuadricula_zonas(distribucion):
    """Presenta las nueve zonas con el mismo orden espacial de la cámara."""
    tabla = pd.DataFrame(
        [
            [
                distribucion["Superior izquierda"],
                distribucion["Superior centro"],
                distribucion["Superior derecha"],
            ],
            [
                distribucion["Centro izquierda"],
                distribucion["Centro"],
                distribucion["Centro derecha"],
            ],
            [
                distribucion["Inferior izquierda"],
                distribucion["Inferior centro"],
                distribucion["Inferior derecha"],
            ],
        ],
        index=["Superior", "Centro", "Inferior"],
        columns=["Izquierda", "Centro", "Derecha"],
    )
    st.dataframe(
        tabla.style.format("{:.2f}%"),
        use_container_width=True,
    )


st.set_page_config(page_title="Sistema POS - Visión Artificial", layout="wide")

st.sidebar.title("Navegación")
opcion = st.sidebar.radio(
    "Selecciona un módulo:",
    ["📊 Resumen del Día", "📷 Control de Cámara"],
)

if opcion == "📊 Resumen del Día":
    st.title("📊 Resumen del Día a Día - Flujo de Clientes")

    try:
        df_logs, df_heatmap = cargar_datos()
    except Exception as exc:
        st.error(f"No se pudo leer la base de datos: {exc}")
        st.stop()

    df_logs = preparar_fechas(df_logs)
    df_heatmap = preparar_fechas(df_heatmap)

    fechas = set()
    if not df_logs.empty:
        fechas.update(df_logs["fecha"].unique())
    if not df_heatmap.empty:
        fechas.update(df_heatmap["fecha"].unique())

    if not fechas:
        st.info("La base de datos está vacía. Inicia la cámara para registrar datos.")
        st.stop()

    fechas_disponibles = sorted(fechas)
    fecha_seleccionada = st.selectbox(
        "Selecciona un día:",
        fechas_disponibles,
        index=len(fechas_disponibles) - 1,
    )

    st.markdown("### Filtro por Horas")
    col_h1, col_h2 = st.columns(2)
    with col_h1:
        hora_inicio = st.time_input("Desde las:", time(0, 0))
    with col_h2:
        hora_fin = st.time_input("Hasta las:", time(23, 59))

    if hora_inicio > hora_fin:
        st.error("La hora inicial no puede ser posterior a la hora final.")
        st.stop()

    logs_fecha = (
        df_logs[df_logs["fecha"] == fecha_seleccionada].copy()
        if not df_logs.empty
        else df_logs
    )
    heatmap_fecha = (
        df_heatmap[df_heatmap["fecha"] == fecha_seleccionada].copy()
        if not df_heatmap.empty
        else df_heatmap
    )

    df_dia = filtrar_por_hora(logs_fecha, hora_inicio, hora_fin)
    df_heatmap_periodo = filtrar_por_hora(heatmap_fecha, hora_inicio, hora_fin)

    entradas = int((df_dia["evento"] == "ENTRADA").sum()) if not df_dia.empty else 0
    salidas = int((df_dia["evento"] == "SALIDA").sum()) if not df_dia.empty else 0
    balance_neto = entradas - salidas

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Entradas", entradas)
    col2.metric("Total Salidas", salidas)
    col3.metric("Tránsito Neto", balance_neto)

    st.subheader("Tráfico por Hora")
    resumen_horas = pd.DataFrame()
    hora_pico = None
    entradas_hora_pico = 0
    hora_pico_str = "No determinada"

    if not df_dia.empty:
        df_dia = df_dia.copy()
        df_dia["hora_numero"] = pd.to_datetime(
            df_dia["hora"], format="%H:%M:%S", errors="coerce"
        ).dt.hour
        resumen_horas = (
            df_dia.groupby(["hora_numero", "evento"])
            .size()
            .unstack(fill_value=0)
        )

        colores = [
            "#00FF00" if columna == "ENTRADA" else "#FF0000"
            for columna in resumen_horas.columns
        ]
        st.bar_chart(resumen_horas, color=colores)

        if "ENTRADA" in resumen_horas.columns and resumen_horas["ENTRADA"].max() > 0:
            hora_pico = int(resumen_horas["ENTRADA"].idxmax())
            entradas_hora_pico = int(resumen_horas["ENTRADA"].max())
            hora_pico_str = (
                f"{hora_pico:02d}:00 hrs ({entradas_hora_pico} entradas)"
            )
    else:
        st.info("No existen entradas o salidas en el rango seleccionado.")

    analisis_calor = analizar_mapa_calor(df_heatmap_periodo)

    st.markdown("---")
    st.subheader("🔥 Distribución del mapa de calor")
    st.caption(
        "Los porcentajes representan actividad visual relativa por zona. "
        "No son temperaturas ni mediciones de sensores térmicos."
    )

    if analisis_calor["disponible"]:
        col_z1, col_z2, col_z3 = st.columns(3)
        col_z1.metric("Zona con mayor actividad", analisis_calor["zona_mayor"])
        col_z2.metric(
            "Concentración relativa",
            f"{analisis_calor['concentracion_mayor']:.2f}%",
        )
        col_z3.metric("Muestras analizadas", analisis_calor["muestras"])
        mostrar_cuadricula_zonas(analisis_calor["distribucion"])
    else:
        st.info(
            "No hay muestras con actividad del mapa de calor para este periodo. "
            "Inicia la cámara y espera al menos cinco segundos."
        )

    st.markdown("---")
    st.subheader("🤖 Análisis generado por el agente Groq")
    st.caption(
        "Groq recibe métricas agregadas de conteo y distribución espacial; "
        "no recibe el video ni la imagen del mapa de calor."
    )

    metricas_agente = {
        "fecha": str(fecha_seleccionada),
        "horaInicio": hora_inicio.strftime("%H:%M"),
        "horaFin": hora_fin.strftime("%H:%M"),
        "entradas": entradas,
        "salidas": salidas,
        "balanceNeto": balance_neto,
        "horaPico": (
            f"{hora_pico:02d}:00" if hora_pico is not None else "No determinada"
        ),
        "entradasHoraPico": entradas_hora_pico,
        "totalEventos": int(len(df_dia)),
        "mapaCalorDisponible": analisis_calor["disponible"],
        "zonaMayorConcentracion": analisis_calor["zona_mayor"],
        "concentracionZonaMayor": analisis_calor["concentracion_mayor"],
        "distribucionZonas": analisis_calor["distribucion"],
        "muestrasMapaCalor": analisis_calor["muestras"],
        "promedioPersonasDetectadas": analisis_calor["promedio_personas"],
        "aforoPromedioEstimado": analisis_calor["aforo_promedio"],
    }

    clave_resumen = json.dumps(metricas_agente, ensure_ascii=False, sort_keys=True)

    if st.button(
        "✨ Generar resumen con Groq",
        type="primary",
        use_container_width=True,
    ):
        with st.spinner("Analizando el flujo y la distribución espacial..."):
            try:
                resumen_ia = generar_resumen_con_agente(metricas_agente)
                st.session_state["resumen_ia"] = resumen_ia
                st.session_state["clave_resumen_ia"] = clave_resumen
            except subprocess.TimeoutExpired:
                st.error("Groq tardó demasiado en responder. Intenta nuevamente.")
            except Exception as exc:
                st.error(f"No se pudo generar el resumen: {exc}")

    resumen_ia = st.session_state.get("resumen_ia")
    clave_guardada = st.session_state.get("clave_resumen_ia")

    if resumen_ia and clave_guardada == clave_resumen:
        st.success("Resumen generado correctamente.")
        st.container(border=True).markdown(resumen_ia)
    elif resumen_ia:
        st.info(
            "Los filtros o datos cambiaron. Genera nuevamente el resumen para actualizarlo."
        )

    st.markdown("---")
    st.subheader("📝 Resumen Ejecutivo del Periodo")

    reporte_texto = (
        "=========================================\n"
        "       REPORTE DE TRÁFICO DIARIO         \n"
        "=========================================\n"
        f"Fecha del reporte: {fecha_seleccionada}\n"
        f"Rango de tiempo:   {hora_inicio} - {hora_fin}\n"
        "-----------------------------------------\n"
        "MÉTRICAS CLAVE:\n"
        f"  - Total Clientes Ingresados: {entradas}\n"
        f"  - Total Clientes Salidos:    {salidas}\n"
        f"  - Balance Neto en Local:     {balance_neto}\n"
        f"  - Hora con más entradas:     {hora_pico_str}\n"
        "-----------------------------------------\n"
        "DISTRIBUCIÓN ESPACIAL:\n"
        f"  - Mapa de calor disponible:  {'Sí' if analisis_calor['disponible'] else 'No'}\n"
        f"  - Zona de mayor actividad:   {analisis_calor['zona_mayor']}\n"
        f"  - Concentración relativa:    {analisis_calor['concentracion_mayor']:.2f}%\n"
        f"  - Muestras analizadas:       {analisis_calor['muestras']}\n"
    )

    if resumen_ia and clave_guardada == clave_resumen:
        reporte_texto += (
            "-----------------------------------------\n"
            "RESUMEN GENERADO POR GROQ:\n"
            f"{resumen_ia}\n"
        )

    reporte_texto += (
        "-----------------------------------------\n"
        f"Reporte generado el: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        "========================================="
    )

    st.text(reporte_texto)
    st.download_button(
        label="📥 Exportar Resumen a Texto (.txt)",
        data=reporte_texto,
        file_name=f"resumen_trafico_{fecha_seleccionada}.txt",
        mime="text/plain",
    )

    st.markdown("---")
    with st.expander("Ver historial detallado completo"):
        st.markdown("#### Entradas y salidas")
        if df_dia.empty:
            st.info("No hay registros de cruces para el periodo.")
        else:
            st.dataframe(
                df_dia[["fecha", "hora", "track_id", "evento"]],
                use_container_width=True,
            )

        st.markdown("#### Muestras del mapa de calor")
        if df_heatmap_periodo.empty:
            st.info("No hay muestras espaciales para el periodo.")
        else:
            columnas = [
                "fecha",
                "hora",
                "personas_detectadas",
                "aforo_estimado",
                "zona_mayor",
                "concentracion_mayor",
            ]
            st.dataframe(
                df_heatmap_periodo[columnas],
                use_container_width=True,
            )

elif opcion == "📷 Control de Cámara":
    st.title("🔴 Control del Motor de Visión")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Encendido")
        st.info(
            "La cámara detectará personas, contará cruces y guardará una muestra "
            "del mapa de calor cada cinco segundos."
        )
        if st.button("🚀 Iniciar Cámara Nativa", type="primary"):
            try:
                subprocess.Popen(
                    [sys.executable, os.path.join(BASE_DIR, "tracker_principal.py")],
                    cwd=BASE_DIR,
                )
                st.success("¡Cámara iniciada en segundo plano!")
            except Exception as exc:
                st.error(f"Error al iniciar la cámara: {exc}")

    with col2:
        st.subheader("Herramientas en Vivo")
        st.info("Este botón borra el mapa de calor acumulado en la cámara.")
        if st.button("🔄 Reiniciar Mapa de Calor", type="secondary"):
            reset_path = os.path.join(BASE_DIR, "reset_heatmap.flag")
            with open(reset_path, "w", encoding="utf-8") as archivo:
                archivo.write("reset")
            st.success("¡Señal enviada! El mapa se limpiará instantáneamente.")
