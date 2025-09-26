
-----

# Detector de somnolencia y sistema de alerta

Este proyecto utiliza **Visión por Computadora (OpenCV, Dlib)** para monitorear el rostro de un conductor en tiempo real. Cuando se detectan signos de somnolencia (ojos cerrados durante un período prolongado), el sistema activa alarmas sonoras progresivas y envía una alerta de **WhatsApp** al contacto de emergencia.

-----

## Requisitos Previos

Antes de ejecutar el proyecto, asegúrate de tener instalado **Python 3.x** y las herramientas necesarias.

### 1\. Instalación de Dependencias

Todas las librerías de Python necesarias están listadas en el archivo `requirements.txt`. Instálalas en tu entorno virtual ejecutando el siguiente comando:

```bash
pip install -r requirements.txt
```

### 2\. Archivos Esenciales

Necesitas obtener el archivo de predicción de puntos faciales de Dlib y colocarlo en la carpeta correspondiente.

| Archivo | Ubicación requerida | Descripción |
| :--- | :--- | :--- |
| **`shape_predictor_68_face_landmarks.dat`** | **`./recursos/`** | Esencial para que Dlib localice los 68 puntos clave de la cara (ojos, nariz, boca). |

-----

## Pasos para la Configuración

El proyecto tiene dos fases: **Entrenamiento** (opcional, si usas Reconocimiento Facial) y **Detección**.

### Fase 1: Recolección de Datos (Archivo: `captura.py`)

Si usas reconocimiento facial (LBPH), primero debes recolectar las imágenes de los rostros.

1.  **Ejecuta el script de captura:**
    ```bash
    python captura.py
    ```
2.  **Generación Automática:** Este script automáticamente creará la carpeta **`captura/`** y subcarpetas para cada persona definida (`./captura/conductor/`) donde se guardarán los rostros detectados.

### Fase 2: Entrenamiento del Modelo (Archivo: `entrenamiento.py`)

Después de recolectar los datos, genera el modelo de reconocimiento.

1.  **Ejecuta el script de entrenamiento:**
    ```bash
    python entrenamiento.py
    ```
2.  **Generación Automática:** Este script automáticamente creará la carpeta **`modelos/`** y guardará el modelo entrenado (ej. `modeloLBPHFace.xml`) dentro.

### Fase 3: Configuración del Detector

Edita el archivo principal (`DrowsinessDetector.py`) para personalizar la alerta:

| Parámetro | Ejemplo | Nota |
| :--- | :--- | :--- |
| **`phone_number`** | `"+51915915670"` | **Obligatorio** en formato internacional (código de país + número). |
| **`predictor_path`** | Se configura automáticamente para buscar en `./recursos/` | Asegúrate de haber colocado el archivo **`.dat`** allí. |

-----

## ▶️ Instrucciones de Ejecución del Detector

### 1\. Preparación para WhatsApp

Para que la alerta automática funcione (usando `pywhatkit` y `pyautogui`):

  * Ten la **sesión de WhatsApp Web iniciada** en tu navegador por defecto (usualmente Chrome).
  * La pantalla de la computadora **no debe estar bloqueada ni minimizada** cuando se lance la alerta, ya que la automatización de la pulsación de la tecla `Enter` necesita que la ventana esté activa.

### 2\. Ejecutar el Detector

Ejecuta el script principal desde tu terminal:

```bash
python DrowsinessDetector.py
```

### 3\. Monitoreo

  * Se abrirá la ventana de la cámara (`cv2.imshow`).
  * Si el sistema detecta somnolencia (ojos cerrados por **más de 4 segundos**), activará alarmas progresivas y enviará la alerta de WhatsApp.

### 4\. Finalizar

Presiona la tecla **ESC** mientras la ventana de la cámara está activa para detener la ejecución.

-----

## 📂 Estructura del Proyecto

```
tu-proyecto/
├── DrowsinessDetector.py      # Lógica de detección de somnolencia y alertas.
├── captura.py                 # Script para recolectar imágenes de rostros.
├── entrenamiento.py           # Script para entrenar el modelo LBPH.
├── requirements.txt           # Lista de dependencias.
├── README.md                  # Este archivo.
├── recursos/                  # Archivos de configuración (DEBES colocar el .dat aquí)
│   └── shape_predictor_68_face_landmarks.dat 
├── captura/ * # Datos de entrenamiento (creada por captura.py)
│   └── conductor/ 
├── modelos/ * # Modelo LBPH guardado (creada por entrenamiento.py)
│   └── modeloLBPHFace.xml
└── logs/ * # Archivos de log del sistema (creada por los scripts)
    └── drowsiness_YYYYMMDD.log
```