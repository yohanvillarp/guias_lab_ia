# Parte 1 - Captura de imágenes faciales

import cv2
import os
import imutils
from datetime import datetime

def create_directory(path):
    """Crea un directorio si no existe"""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f'Carpeta creada: {path}')

def capture_faces(person_name, max_images=300, min_confidence=1.3):
    """Función principal para capturar rostros"""
    # Configuración de rutas
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, 'captura') 
    person_path = os.path.join(data_path, person_name)
    create_directory(person_path)

    # Inicialización de la cámara y el clasificador
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades +
        'haarcascade_frontalface_default.xml')

    if not cap.isOpened():
        print("Error: No se pudo acceder a la cámara")
        return

    count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error al capturar el frame")
                break

            frame = imutils.resize(frame, width=640)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            aux_frame = frame.copy()

            faces = face_classifier.detectMultiScale(
                gray,
                scaleFactor=min_confidence,
                minNeighbors=5,
                minSize=(30, 30)
            )

            for (x, y, w, h) in faces:
                # Dibujar rectángulo y mostrar contador
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f'Imagenes: {count}', (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Procesar y guardar el rostro
                face = aux_frame[y:y+h, x:x+w]
                face = cv2.resize(face, (150, 150), interpolation=cv2.INTER_CUBIC)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f'rostro_{count}_{timestamp}.jpg'
                cv2.imwrite(os.path.join(person_path, filename), face)
                count += 1

            cv2.imshow('Captura de Rostros', frame)

            # Salir o si alcanza el máximo de imágenes
            key = cv2.waitKey(1)
            if key == 27 or count >= max_images:  # 27 = tecla ESC
                break

    except Exception as e:
        print(f"Error durante la captura: {str(e)}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"Captura finalizada. Se guardaron {count} imágenes")

if __name__ == "__main__":
    capture_faces("conductor")

