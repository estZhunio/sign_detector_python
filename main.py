from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
import cv2
import os

# Función para cargar las clases desde labels.txt
def cargar_clases(ruta_label_txt):
    try:
        with open(ruta_label_txt, 'r') as file:
            clases = [line.strip().split(" ", 1)[1] for line in file]
        print(clases)
        return clases
    except Exception as e:
        print(f"Error al cargar clases: {e}")
        return []

# Rutas del modelo y etiquetas
ruta_modelo_h5 = "models/keras_model.h5"
ruta_modelo_guardado = "models/saved_model"
ruta_label_txt = "models/labels.txt"

# Cargar las clases
clases = cargar_clases(ruta_label_txt)

# Verificar si el modelo en formato SavedModel ya existe
if not os.path.exists(ruta_modelo_guardado):
    try:
        print("Convirtiendo modelo .h5 a formato SavedModel...")
        model_h5 = tf.keras.models.load_model(ruta_modelo_h5)
        model_h5.save(ruta_modelo_guardado, save_format="tf")
        print("Conversión completa.")
    except Exception as e:
        raise RuntimeError(f"Error al convertir modelo: {e}")

# Cargar el modelo
try:
    model = tf.keras.models.load_model(ruta_modelo_guardado)
    print("Modelo cargado correctamente.")
except Exception as e:
    raise RuntimeError(f"Error al cargar el modelo: {e}")

# Inicializar FastAPI
app = FastAPI()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        print(f"Nombre del archivo: {file.filename}")
        print(f"Tipo de contenido: {file.content_type}")

        # Verificar si el archivo es una imagen
        if file.content_type not in ["image/jpeg", "image/png"]:
            raise HTTPException(status_code=400, detail="Formato de archivo no soportado")

        # Leer la imagen
        img_bytes = await file.read()
        img_np = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="No se pudo decodificar la imagen")

        # Preprocesar la imagen
        img_resized = cv2.resize(img, (224, 224))  # Ajusta el tamaño según tu modelo
        img_blurred = cv2.GaussianBlur(img_resized, (5, 5), 0)  # 🔹 Aplica suavizado Gaussiano
        img_normalized = img_blurred / 255.0
        img_expanded = np.expand_dims(img_normalized, axis=0)

        # Realizar la predicción
        prediccion = model.predict(img_expanded)
        confianza = prediccion.max()  # 🔹 Obtiene la confianza más alta
        indice_clase = np.argmax(prediccion)

        if indice_clase >= len(clases):
            raise HTTPException(status_code=500, detail="Índice de predicción fuera de rango")

        if confianza >= 0.97:  # 🔥 Nuevo umbral de confianza
            clase_predicha = clases[indice_clase]
        else:
            clase_predicha = ""

        return JSONResponse(content={"prediccion": clase_predicha, "confianza": float(confianza)})

    except Exception as e:
        print(f"Error en la predicción: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {e}")
