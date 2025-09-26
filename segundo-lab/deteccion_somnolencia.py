import os
import time
import logging
import threading
from datetime import datetime

import cv2
import dlib 
import numpy as np
from scipy.spatial import distance as dist
import pywhatkit as kit
import winsound

class DrowsinessDetector:
    def __init__(self, predictor_path: str, phone_number: str,
                 ear_threshold: float = 0.25, alert_cooldown: int = 60, 
                 beep_cooldown: float = 1.0):
        """
        Detector de somnolencia optimizado
        Args: 
            predictor_path (str): Ruta al archivo shape_predictor_68_face_landmarks.dat
            phone_number (str): Número de teléfono para enviar alertas de WhatsApp en formato internacional (ejemplo: "+34123456789")
            ear_threshold (float): Umbral del Eye Aspect Ratio (EAR) para detectar ojos cerrados
            alert_cooldown (int): Tiempo en segundos entre alertas de WhatsApp
            beep_cooldown (float): Tiempo en segundos entre pitidos de alerta sonora
        """
        #consecutive_frames: int = 20,
        #alert_sound_path: str = "alert.wav",
        #log_file: str = "drowsiness_log.txt"):

        self.ear_threshold = ear_threshold
        self.phone_number = phone_number
        self.alert_cooldown = alert_cooldown
        self.beep_cooldown = beep_cooldown

        self.start_time = None
        self.last_alert_time = 0
        self.last_beep_time = 0
        self.alert_active = False

        # Frecuencias progresivas por segundo (1...4+)
        self.beep_frequencies = {1: 500, 2: 750, 3: 1000, 4: 1500}
        
        # Inicialización del detector y predictor de dlib
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)

        self.setup_logging()
    
    def setup_logging(self):
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(os.path.join(log_dir, f"drowsiness_{datetime.now().strftime('%Y%m%d')}.log")),
                logging.StreamHandler()
            ]
        )

    @staticmethod
    def calculate_ear(eye: np.ndarray) -> float:
        """
        Calcula el Eye Aspect Ratio (EAR)
        Args:
            eye (np.ndarray): Coordenadas de los puntos del ojo
        Returns:
            float: Valor del EAR
        """
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        
        if C == 0:
            return 0.0
        return (A + B) / (2.0 * C)
    
    def _beep_async(self, freq: int, dur: int):
        """Ejecuta winsound. Beep en hilo para no bloquear el flujo principal"""
        threading.Thread(target=winsound.Beep, args=(freq, dur), daemon=True).start()

    def sound_alarm(self, frequency: int = 1000, duration_ms: int = 300):
        """Reproduce un sonido de alerta"""
        now = time.time()
        if now - self.last_beep_time >= self.beep_cooldown:
            self._beep_async(frequency, duration_ms)
            self.last_beep_time = now
            logging.warning(f"Alerta sonora: {frequency}Hz {duration_ms}ms")    

    def send_whatsapp_alert(self):
        """Envia mensaje WhatsApp con control de cooldown para enviar spam."""
        now = time.time()
        if now - self.last_alert_time < self.alert_cooldown:
            logging.info("Alert cooldown activo, no se envía WhatsApp.")
            return
        try:
            msg = "ALERTA: Conductor presenta ojos cerrados por >4s. Revise al conductor"
            # pywhatkit.sendwhatmsg_instantly() no tiene parámetros para cerrar la pestaña.
            # Se usa `sendwhatmsg` para programar el envío.
            current_hour = datetime.now().hour
            current_minute = datetime.now().minute + 1
            kit.sendwhatmsg(self.phone_number, msg, current_hour, current_minute, wait_time=10, close_time=3)
            self.last_alert_time = now
            logging.info(f"Alerta WhatsApp enviada a {self.phone_number}")
        except Exception as e:
            logging.error(f"Error enviando alerta WhatsApp: {e}")

    def sound_progressive_alarm(self, elapsed: float):
        """Reproduce pitidos progresivos según segundos de somnolencia"""
        if elapsed < 1.0:
            return
        now = time.time()
        if now - self.last_beep_time < self.beep_cooldown:
            return
        
        seconds = min(int(elapsed), 4) # 1 ...4
        freq = self.beep_frequencies.get(seconds, self.beep_frequencies[4])
        duration = 200 + (seconds - 1) * 100 # aumentar duracion ligeramente
        self._beep_async(freq, duration)
        self.last_beep_time = now
        logging.info(f"Beep progresivo: {seconds}s -> {freq}Hz {duration}ms")

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Procesa un frame para detectar somnolencia
        Args:
            frame (np.ndarray): Frame de video
        Returns:
            np.ndarray: Frame procesado con anotaciones
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray, 0)

        for face in faces:
            shape = self.predictor(gray, face)
            coords = np.array([[p.x, p.y] for p in shape.parts()])

            left_eye = coords[36:42]
            right_eye = coords[42:48]

            left_ear = self.calculate_ear(left_eye)
            right_ear = self.calculate_ear(right_eye)
            ear = (left_ear + right_ear) / 2.0

            self.draw_eyes(frame, left_eye, right_eye)
            
            if ear < self.ear_threshold:
                if self.start_time is None:
                    self.start_time = time.time()
                    logging.info("Ojos cerrados detectados, iniciando temporizador.")

                elapsed = time.time() - self.start_time
                # Alertas sonoras cada segundo
                self.sound_progressive_alarm(elapsed)

                # Mostrar tiempo de ojos cerrados
                self.draw_alert_status(frame, elapsed)

                # a partir de 4s: alerta mantenida + whatsapp
                if elapsed >= 4.0:
                    if not self.alert_active:
                        logging.warning("Alerta de somnolencia activada mas de 4s.")
                        self.alert_active = True
                        # alarma continua de mayor tono
                        self.sound_alarm(frequency=self.beep_frequencies[4], duration_ms=400)
                        self.draw_alert(frame)
                        self.send_whatsapp_alert()

            else:
                # resetear estado
                if self.start_time is not None:
                    logging.info(f"Ojos abiertos, reseteando estado de alerta después de {time.time() - self.start_time:.2f}s.")
                    self.start_time = None
                    self.alert_active = False
        return frame

    def draw_eyes(self, frame: np.ndarray, left_eye: np.ndarray, right_eye: np.ndarray):
        """Dibuja los contornos de los ojos en el frame"""
        
        for eye in (left_eye, right_eye):
            hull = cv2.convexHull(eye)
            cv2.drawContours(frame, [hull], -1, (0, 255, 0), 1)
            for (x, y) in eye:
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

    def draw_alert(self, frame: np.ndarray):
        """Dibuja mensaje de alerta critica en pantalla."""
        h, w = frame.shape[:2]
        cv2.putText(frame, "ALERTA SOMNOLENCIA!", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.rectangle(frame, (5, 5), (w - 5, 50), (0, 0, 255), 2)

    def draw_alert_status(self, frame: np.ndarray, elapsed_time: float):
        """Dibuja el estado de alerta en el frame"""
        cv2.putText(frame, f"OJOS CERRADOS {elapsed_time:.1f}s", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        level = min(int(elapsed_time), 4)
        cv2.putText(frame, f"Nivel Alerta: {level}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # barra de progreso (0..4s -> 0..200px)
        bar_len = int(min(elapsed_time/4.0, 1.0) * 200)
        cv2.rectangle(frame, (10, 100), (10 + 200, 120), (50, 50, 50), 1)
        if bar_len > 0:
            cv2.rectangle(frame, (10, 100), (10 + bar_len, 120), (0, 0, 255), -1)

    def run(self, cam_index: int = 0):
        """Bucle principal de captura y deteccion. """
        cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            logging.error("No se pudo abrir la cámara con índice")
            return

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logging.error("No se pudo leer el frame de la cámara.")
                    break

                frame = cv2.resize(frame, (640, int(frame.shape[0] * 640 / frame.shape[1])))
                processed = self.process_frame(frame)

                cv2.imshow("Monitoreo Somnolencia", processed)

                key = cv2.waitKey(1) & 0xFF
                if key == 27: # Esc
                    break
        except Exception as e:
            logging.error(f"Error en el bucle principal: {e}")

        finally:
            cap.release()
            cv2.destroyAllWindows()
            logging.info("Detector finalizado.")
    
def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    relative_path = os.path.join(base_dir, "recursos", "shape_predictor_68_face_landmarks.dat")
    predictor_path = relative_path
    phone_number = "+51915915670"
    detector = DrowsinessDetector(predictor_path, phone_number)
    detector.run()

if __name__ == "__main__":
    main()    