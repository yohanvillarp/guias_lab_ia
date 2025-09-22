# módulos importados

import cv2
import numpy as np
import pygame
import threading
import time
import sys

# Constantes - visualización

TITLE_FRAME: str = "Detección de fuego"
VIDEO_WIDTH: int = 1000
VIDEO_HEIGHT: int = 600
BRIGHTNESS: int = 0
CONTRAST: float = 1.2
GAUSSIAN_KERNEL: tuple = (15, 15)
MIN_FIRE_AREA: int = 2000


# constantes - audio

ALARM_SOUND: str = "fire-alarm.mp3"
PRE_ALARM_SOUND: str = "pre-alarm.mp3"


# Umbrales para la alerta escalonada

ALERTA_THRESHOLD = 5
ALARMA_THRESHOLD = 15

# Rangos de color HSV

# Rojo intenso
FIRE_LOWER1 = np.array([0, 120, 220], dtype="uint8")
FIRE_UPPER1 = np.array([10, 255, 255], dtype="uint8")

# Naranja brillante
FIRE_LOWER2 = np.array([11, 180, 220], dtype="uint8")
FIRE_UPPER2 = np.array([25, 255, 255], dtype="uint8")

# Amarillo muy luminoso
FIRE_LOWER3 = np.array([26, 200, 230], dtype="uint8")
FIRE_UPPER3 = np.array([35, 255, 255], dtype="uint8")


# Variables de estado

estado_alarma: bool = False
alarma_hilo: threading.Thread = None
alarma_activa: threading.Event = threading.Event()
deteccion_continua: int = 0
nivel_alarma: str = "OFF"


# Funciones de audio

def audio_loop():
    try:
        pygame.mixer.init()
        current_sound_loaded = None
        while alarma_activa.is_set():
            if nivel_alarma == "ALERT" and current_sound_loaded != "ALERT":
                pygame.mixer.music.stop()
                pygame.mixer.music.load(PRE_ALARM_SOUND)
                pygame.mixer.music.play(-1)
                current_sound_loaded = "ALERT"
            elif nivel_alarma == "ALARM" and current_sound_loaded != "ALARM":
                pygame.mixer.music.stop()
                pygame.mixer.music.load(ALARM_SOUND)
                pygame.mixer.music.play(-1)
                current_sound_loaded = "ALARM"
            elif nivel_alarma == "OFF" and current_sound_loaded != "OFF":
                pygame.mixer.music.stop()
                current_sound_loaded = "OFF"
            time.sleep(0.5)
    except pygame.error as e:
        print(f"Error en el hilo de audio: {e}")
    finally:
        if pygame.mixer.get_init():
            pygame.mixer.music.stop()
            pygame.mixer.quit()


def start_audio_thread():
    global alarma_hilo
    if not alarma_hilo or not alarma_hilo.is_alive():
        alarma_activa.set()
        alarma_hilo = threading.Thread(target=audio_loop)
        alarma_hilo.start()
        print("Evento: Hilo de audio iniciado.")


def stop_audio_thread():
    global alarma_hilo, nivel_alarma
    if alarma_hilo and alarma_hilo.is_alive():
        alarma_activa.clear()
        nivel_alarma = "OFF"
        alarma_hilo.join()
        print("Evento: Hilo de audio detenido.")


# Procesamiento de imagen

def preprocess_frame(frame):
    frame = cv2.resize(frame, (VIDEO_WIDTH, VIDEO_HEIGHT))
    return cv2.convertScaleAbs(frame, alpha=CONTRAST, beta=BRIGHTNESS)


def apply_blur_and_hsv(frame):
    blur = cv2.GaussianBlur(frame, GAUSSIAN_KERNEL, 0)
    return cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)


# Detección de fuego

def detectar_fuego(hsv, frame):
    mask1 = cv2.inRange(hsv, FIRE_LOWER1, FIRE_UPPER1)
    mask2 = cv2.inRange(hsv, FIRE_LOWER2, FIRE_UPPER2)
    mask3 = cv2.inRange(hsv, FIRE_LOWER3, FIRE_UPPER3)
    mask = cv2.bitwise_or(mask1, mask2, mask3)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    fire_detected = False
    for contour in contours:
        if cv2.contourArea(contour) > MIN_FIRE_AREA:
            fire_detected = True
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "fuego detectado", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
    return fire_detected


# Lógica de alarma

def manejar_evento_alarma(fire_detected):
    global estado_alarma, deteccion_continua, nivel_alarma

    if fire_detected:
        deteccion_continua += 1
        if deteccion_continua >= ALARMA_THRESHOLD and nivel_alarma != "ALARM":
            nivel_alarma = "ALARM"
            if not alarma_hilo or not alarma_hilo.is_alive():
                start_audio_thread()
            print("Evento: Nivel de alarma Aumentado a ALARMA.")
        elif deteccion_continua >= ALERTA_THRESHOLD and nivel_alarma == "OFF":
            nivel_alarma = "ALERT"
            if not alarma_hilo or not alarma_hilo.is_alive():
                start_audio_thread()
            print("Evento: Nivel de alarma Aumentado a ALERTA.")
    else:
        deteccion_continua = 0
        if nivel_alarma != "OFF":
            nivel_alarma = "OFF"
            print("Evento: Nivel de alarma restablecido a OFF.")


# Bucle principal

def main():
    video = cv2.VideoCapture(0)
    if not video.isOpened():
        print("Error: No se pudo abrir la cámara.")
        return

    try:
        while True:
            ret, frame = video.read()
            if not ret:
                print("Error: No se pudo leer el fotograma.")
                break

            frame = preprocess_frame(frame)
            hsv = apply_blur_and_hsv(frame)

            fire_detected = detectar_fuego(hsv, frame)
            manejar_evento_alarma(fire_detected)

            cv2.imshow(TITLE_FRAME, frame)

            key = cv2.waitKey(1)
            if key == 27:
                break
            if cv2.getWindowProperty(TITLE_FRAME, cv2.WND_PROP_VISIBLE) < 1:
                break
    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")
    finally:
        print("Finalizando aplicación...")
        video.release()
        cv2.destroyAllWindows()
        stop_audio_thread()
        sys.exit()


if __name__ == "__main__":
    main()