# Parte 2 - entrenamiento del modelo LBPH

import cv2
import os
import numpy as np
import logging 
from datetime import datetime

class FaceModelTrainer:
    def __init__(self, data_path, model_path):
        """
        Inicializa el entrenador del modelo facial
        Args:
            data_path (str): Ruta a los datos de entrenamiento
            model_path (str): Ruta donde se guardará el modelo
        """
        self.data_path = data_path
        self.model_path = model_path
        self.setup_logging()

    def setup_logging(self):
        """Configura el sistema de logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'training_{datetime.now().strftime("%Y%m%d")}.log'),
                logging.StreamHandler()
            ]
        )
        
    def load_training_data(self):
        """Carga las imagenes y etiquetas para el entrenamiento"""
        faces_data = []
        labels = []
        label = 0
        
        try:
            people_list = os.listdir(self.data_path)
            
            if not people_list:
                raise ValueError(
                    "No se encontraron carpetas de personas en la ruta especificada."
                )
            
            # --- CORRECCIÓN CLAVE 2: Indentación de logging ---
            # Este logging debe ir FUERA del 'if not people_list'
            logging.info(f"Personas encontradas: {people_list}") 

            for person_name in people_list:
                person_path = os.path.join(self.data_path, person_name)
                
                if not os.path.isdir(person_path):
                    continue
                face_count = 0 
                
                for filename in os.listdir(person_path):
                    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        continue

                    image_path = os.path.join(person_path, filename)
                    # Cargar la imagen en escala de grises (0)
                    image = cv2.imread(image_path, 0)

                    if image is None:
                        logging.warning(f"No se pudo cargar la imagen: {image_path}. Se omitirá.")
                        continue # Si no se carga, saltamos la imagen para evitar errores en cv2.face.LBPHFaceRecognizer.train()

                    # Si la imagen es válida:
                    faces_data.append(image)
                    labels.append(label)
                    face_count += 1

                logging.info(f"Procesadas {face_count} imágenes para {person_name}")
                label += 1

            return faces_data, labels

        except Exception as e:
            logging.error(f"Error al cargar los datos: {str(e)}")
            raise 

    def train_model(self):
        """Entrena el modelo LBPH con las imágenes cargadas"""
        try:
            faces_data, labels = self.load_training_data()

            if not faces_data:
                raise ValueError(
                    "No se encontraron datos válidos para entrenar"
                )
            logging.info("Iniciando entrenamiento del modelo...") 

            face_recognizer = cv2.face.LBPHFaceRecognizer_create()
            # np.array(labels) asegura que las etiquetas sean un numpy array, necesario para el entrenamiento
            face_recognizer.train(faces_data, np.array(labels))

            # Crear directorio si no existe
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

            face_recognizer.write(self.model_path)
            logging.info(f"Modelo guardado exitosamente en {self.model_path}")

            return True

        except Exception as e:
            logging.error(f"Error durante el entrenamiento: {str(e)}")
            return False
            
def main():
    # 1. Obtener el directorio base (donde se ejecuta este script)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # 2. Rutas relativas para datos y modelo
    # Ruta de los datos: ./captura
    data_path = os.path.join(base_dir, 'captura')
    # Ruta del modelo: ./modelos/modeloLBPHFace.xml
    model_path = os.path.join(base_dir, 'modelos', 'modeloLBPHFace.xml')
    # --------------------------------------------------------

    trainer = FaceModelTrainer(data_path, model_path)
    trainer.train_model()

if __name__ == "__main__":
    main()