from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
import cv2
import base64
import os
import logging
import time
from typing import Optional

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Función para cargar las clases desde labels.txt
def cargar_clases(ruta_label_txt):
    try:
        with open(ruta_label_txt, 'r') as file:
            clases = [line.strip().split(" ", 1)[1] for line in file]
        logger.info(f"Clases cargadas: {clases}")
        return clases
    except Exception as e:
        logger.error(f"Error al cargar clases: {e}")
        return []


# Rutas del modelo y etiquetas
ruta_modelo_h5 = "models/keras_model.h5"
ruta_modelo_guardado = "models/saved_model"
ruta_label_txt = "models/labels.txt"

# Cargar las clases
clases = cargar_clases(ruta_label_txt)

# Verificar y cargar el modelo
if not os.path.exists(ruta_modelo_guardado):
    try:
        logger.info("Convirtiendo modelo .h5 a formato SavedModel...")
        model_h5 = tf.keras.models.load_model(ruta_modelo_h5)
        model_h5.save(ruta_modelo_guardado, save_format="tf")
        logger.info("Conversión completa.")
    except Exception as e:
        raise RuntimeError(f"Error al convertir modelo: {e}")

try:
    model = tf.keras.models.load_model(ruta_modelo_guardado)
    logger.info("Modelo cargado correctamente.")
except Exception as e:
    raise RuntimeError(f"Error al cargar el modelo: {e}")


class SignProcessor:
    def __init__(self):
        self.last_process_time = time.time()
        self.processing_interval = 0.1  # 100ms entre procesamientos
        self.confidence_threshold = 0.93  # Umbral de confianza

    async def process_frame(self, frame_data: bytes) -> dict:
        # Control de tasa de procesamiento
        current_time = time.time()
        if current_time - self.last_process_time < self.processing_interval:
            return {"prediccion": "", "confianza": 0.0}

        try:
            # Decodificar frame
            nparr = np.frombuffer(base64.b64decode(frame_data), np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                return {"prediccion": "", "confianza": 0.0}

            # Preprocesar imagen (igual que en tu código original)
            img_resized = cv2.resize(img, (224, 224))
            img_blurred = cv2.GaussianBlur(img_resized, (5, 5), 0)
            img_normalized = img_blurred / 255.0
            img_expanded = np.expand_dims(img_normalized, axis=0)

            # Realizar predicción
            prediccion = model.predict(img_expanded, verbose=0)
            confianza = float(prediccion.max())
            indice_clase = np.argmax(prediccion)

            # Verificar umbral de confianza
            if confianza >= self.confidence_threshold and indice_clase < len(clases):
                clase_predicha = clases[indice_clase]
            else:
                clase_predicha = ""

            self.last_process_time = current_time

            return {
                "prediccion": clase_predicha,
                "confianza": confianza,
                "processing_time": time.time() - current_time
            }

        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return {"prediccion": "", "confianza": 0.0}


processor = SignProcessor()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("Nueva conexión WebSocket establecida")
    print("WebSocket conectado!")

    try:
        while True:
            # Recibir frame
            data = await websocket.receive_json()
            logger.info(f"Datos recibidos del cliente: {data}")

            # Procesar frame
            result = await processor.process_frame(data['frame'])

            # Enviar resultado
            logger.info(f"Respuesta enviada al cliente: {result}")
            await websocket.send_json(result)

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await websocket.close()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)