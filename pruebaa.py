import cv2
import mediapipe as mp
import time
from tensorflow.keras.models import load_model
import numpy as np
def convert_label(x):
	if x == '0':
		return 'A'
	elif x == '1':
		return 'B'
	elif x == '2':
		return 'C'
	elif x == '3':
		return 'D'
	elif x == '4':
		return 'E'
	elif x == '5':
		return 'F'
	elif x == '6':
		return 'G'
	elif x == '7':
		return 'H'
	elif x == '8':
		return 'I'
	elif x == '9':
		return 'J'
	elif x == '10':
		return 'K'
	elif x == '11':
		return 'L'
	elif x == '12':
		return 'M'
	elif x == '13':
		return 'N'
	elif x == '14':
		return 'O'
	elif x == '15':
		return 'P'
	elif x == '16':
		return 'Q'
	elif x == '17':
		return 'R'
	elif x == '18':
		return 'S'
	elif x == '19':
		return 'T'
	elif x == '20':
		return 'U'
	elif x == '21':
		return 'V'
	elif x == '22':
		return 'W'
	elif x == '23':
		return 'X'
	elif x == '24':
		return 'Y'
	elif x == '25':
		return 'Z'
# Carga el modelo
model_path = "./prueba2.keras"
model = load_model(model_path)
print("Modelo cargado")
# Inicializar el objeto de detección de manos de Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing.style = mp.solutions.drawing_styles.get_default_hand_landmarks_style()
hands = mp_hands.Hands()

# Inicializar la cámara
cap = cv2.VideoCapture(0)

while True:
    # Capturar el frame de la cámara
    ret, frame = cap.read()

    # Convertir el frame a RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detectar las manos en el frame
    results = hands.process(frame_rgb)

    # Si se detectaron manos
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]

            # Dibujar landmarks de la mano en el frame
        mp_drawing.draw_landmarks(frame, 
                                      hand_landmarks, 
                                      mp_hands.HAND_CONNECTIONS)
        

    # Mostrar el frame con los puntos de referencia de la mano
    cv2.imshow('Hand Gestures', frame)

    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    img = cv2.resize(frame, (64, 64))  # Asegúrate de redimensionar según el tamaño esperado por tu modelo
    img = img / 255.0  # Normaliza los valores de píxeles 
    img = np.expand_dims(img, axis=0)
    prediction=model.predict(img)

    print("Prediction", convert_label(str(np.argmax(prediction))))
    time.sleep(1)

# Liberar los recursos
cap.release()
cv2.destroyAllWindows()