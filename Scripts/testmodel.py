
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import mediapipe as mp
import cv2
import numpy as np
import time
# Carga el modelo
model_path = "./prueba3.keras"
model = load_model(model_path)
print("Modelo cargado")


# Inicializar Mediapipe Hand module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing.style = mp.solutions.drawing_styles.get_default_hand_landmarks_style()
hands = mp_hands.Hands()

# Inicializar webcam
cap = cv2.VideoCapture(0)
while True:
    # Leer frame de la webcam
    
        ret, frame = cap.read()
        
        # Convertir frame a RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imshow('a',frame)
        cv2.waitKey(1)
        # Procesar frame con Mediapipe Hand module
        results = hands.process(frame_rgb)
    
        if results.multi_hand_landmarks:
            # Asignar landmarks de la mano
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Dibujar landmarks de la mano en el frame
            mp_drawing.draw_landmarks(frame_rgb, 
                                      hand_landmarks, 
                                      mp_hands.HAND_CONNECTIONS)
        
        # Mostrar frame
        # cv2.imshow('Hand Tracking', cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
        img = cv2.resize(frame_rgb, (64, 64))  # Asegúrate de redimensionar según el tamaño esperado por tu modelo
        img = img / 255.0  # Normaliza los valores de píxeles 
        img = np.expand_dims(img, axis=0)
        prediction=model.predict(img)
        predicted_class = np.argmax(prediction)
        print("Prediction", predicted_class)
