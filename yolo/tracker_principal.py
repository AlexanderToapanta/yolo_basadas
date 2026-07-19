import os
import time

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

import database as db


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "yolo11n.pt")
RESET_FLAG = os.path.join(BASE_DIR, "reset_heatmap.flag")

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
INTERVALO_GUARDADO_SEGUNDOS = 5

NOMBRES_ZONAS = {
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

ORDEN_ZONAS = (
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


def calcular_distribucion_zonas(heatmap_normalizado):
    """
    Divide el mapa de calor en una cuadrícula 3x3 y calcula qué porcentaje de
    la actividad visual acumulada corresponde a cada zona.
    """
    alto, ancho = heatmap_normalizado.shape
    limites_y = np.linspace(0, alto, 4, dtype=int)
    limites_x = np.linspace(0, ancho, 4, dtype=int)

    energia_por_zona = {}
    indice = 0

    for fila in range(3):
        for columna in range(3):
            zona = ORDEN_ZONAS[indice]
            indice += 1

            region = heatmap_normalizado[
                limites_y[fila] : limites_y[fila + 1],
                limites_x[columna] : limites_x[columna + 1],
            ]
            energia_por_zona[zona] = float(region.astype(np.float64).sum())

    energia_total = sum(energia_por_zona.values())
    if energia_total <= 0:
        porcentajes = {zona: 0.0 for zona in ORDEN_ZONAS}
        return porcentajes, "Sin actividad", 0.0

    porcentajes = {
        zona: round((energia / energia_total) * 100.0, 2)
        for zona, energia in energia_por_zona.items()
    }
    zona_mayor_clave = max(porcentajes, key=porcentajes.get)
    return (
        porcentajes,
        NOMBRES_ZONAS[zona_mayor_clave],
        porcentajes[zona_mayor_clave],
    )


def dibujar_cuadricula_zonas(frame, zona_mayor):
    """Dibuja la división 3x3 utilizada para calcular las zonas."""
    x1 = FRAME_WIDTH // 3
    x2 = (FRAME_WIDTH * 2) // 3
    y1 = FRAME_HEIGHT // 3
    y2 = (FRAME_HEIGHT * 2) // 3

    cv2.line(frame, (x1, 0), (x1, FRAME_HEIGHT), (180, 180, 180), 1)
    cv2.line(frame, (x2, 0), (x2, FRAME_HEIGHT), (180, 180, 180), 1)
    cv2.line(frame, (0, y1), (FRAME_WIDTH, y1), (180, 180, 180), 1)
    cv2.line(frame, (0, y2), (FRAME_WIDTH, y2), (180, 180, 180), 1)

    cv2.putText(
        frame,
        f"Zona de mayor actividad: {zona_mayor}",
        (20, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )


def main():
    db.init_db()

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"No se encontró el modelo YOLO en: {MODEL_PATH}")

    modelo = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la cámara.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    heatmap_acc = np.zeros((FRAME_HEIGHT, FRAME_WIDTH), dtype=np.float32)

    punto_inicio = sv.Point(FRAME_WIDTH // 2, 0)
    punto_fin = sv.Point(FRAME_WIDTH // 2, FRAME_HEIGHT)
    line_zone = sv.LineZone(start=punto_inicio, end=punto_fin)
    box_annotator = sv.BoxAnnotator(thickness=1)

    tiempo_anterior = time.time()
    ultimo_guardado = 0.0

    print(
        "Iniciando tracker con conteo y análisis 3x3 del mapa de calor. "
        "Presiona ESC para salir."
    )

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if os.path.exists(RESET_FLAG):
                heatmap_acc.fill(0.0)
                try:
                    os.remove(RESET_FLAG)
                except OSError:
                    pass

            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

            ahora = time.time()
            diferencia = max(ahora - tiempo_anterior, 1e-6)
            fps = 1.0 / diferencia
            tiempo_anterior = ahora

            resultados = modelo.track(
                frame,
                classes=[0],
                conf=0.5,
                persist=True,
                tracker="bytetrack.yaml",
                imgsz=480,
                verbose=False,
            )[0]
            detections = sv.Detections.from_ultralytics(resultados)

            # Desvanecimiento gradual del rastro acumulado.
            heatmap_acc = np.clip(heatmap_acc - 0.003, 0.0, 1.0)

            if resultados.boxes is not None:
                for box in resultados.boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = box
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    cv2.circle(
                        heatmap_acc,
                        (cx, cy),
                        radius=30,
                        color=1.0,
                        thickness=-1,
                    )

            heatmap_suavizado = cv2.GaussianBlur(heatmap_acc, (51, 51), 0)
            heatmap_normalizado = np.clip(
                heatmap_suavizado * 255, 0, 255
            ).astype(np.uint8)
            heatmap_color = cv2.applyColorMap(
                heatmap_normalizado, cv2.COLORMAP_JET
            )

            mascara = heatmap_normalizado > 15
            frame_anotado = frame.copy()
            frame_anotado[mascara] = cv2.addWeighted(
                frame, 0.5, heatmap_color, 0.5, 0
            )[mascara]

            if len(detections) > 0 and detections.tracker_id is not None:
                crossed_in, crossed_out = line_zone.trigger(detections=detections)

                for indice, cruzo in enumerate(crossed_in):
                    if cruzo:
                        db.insert_log(int(detections.tracker_id[indice]), "ENTRADA")

                for indice, cruzo in enumerate(crossed_out):
                    if cruzo:
                        db.insert_log(int(detections.tracker_id[indice]), "SALIDA")

            aforo_actual = int(line_zone.in_count - line_zone.out_count)
            personas_detectadas = int(len(detections))
            zonas, zona_mayor, concentracion_mayor = calcular_distribucion_zonas(
                heatmap_normalizado
            )

            if ahora - ultimo_guardado >= INTERVALO_GUARDADO_SEGUNDOS:
                try:
                    db.insert_heatmap_snapshot(
                        zonas=zonas,
                        personas_detectadas=personas_detectadas,
                        aforo_estimado=aforo_actual,
                        zona_mayor=zona_mayor,
                        concentracion_mayor=concentracion_mayor,
                    )
                    ultimo_guardado = ahora
                except Exception as error:
                    print(f"No se pudo guardar la muestra del mapa de calor: {error}")

            frame_anotado = box_annotator.annotate(
                frame_anotado, detections=detections
            )
            cv2.line(
                frame_anotado,
                (punto_inicio.x, punto_inicio.y),
                (punto_fin.x, punto_fin.y),
                (255, 255, 255),
                2,
            )
            dibujar_cuadricula_zonas(frame_anotado, zona_mayor)

            cv2.putText(
                frame_anotado,
                f"FPS: {int(fps)}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame_anotado,
                f"AFORO: {aforo_actual}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                frame_anotado,
                f"In: {line_zone.in_count} | Out: {line_zone.out_count}",
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            cv2.imshow("Sistema POS - Vision Artificial", frame_anotado)

            tecla = cv2.waitKey(1)
            if tecla == 27:
                break
            if tecla == ord("c"):
                heatmap_acc.fill(0.0)

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
