import cv2
import mediapipe as mp
import os
import time
# Crear directorio para almacenar el dataset
dataset_dir = './dataset/test'
if not os.path.exists(dataset_dir):
    print("Creando directorio para almacenar el dataset")
    os.makedirs(dataset_dir)

# Inicializar Mediapipe Hand module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing.style = mp.solutions.drawing_styles.get_default_hand_landmarks_style()
hands = mp_hands.Hands()

# Inicializar webcam
cap = cv2.VideoCapture(0)

# Definir las letras del abecedario
letras = 'abcdefghijklmnopqrstuvwxyz1'

# Capturar 30 frames de cada letra
for letra in letras:
    count=  0
    if (letra=='1'):
        letra="space"
    print("Letra: ", letra)
    letra_dir = os.path.join(dataset_dir, letra)
    if not os.path.exists(letra_dir):
        os.makedirs(letra_dir)
    
    time.sleep(10)
    print("Capturando  frames de la letra ", letra)
    start_time = time.time()
    while time.time() - start_time < 2:
        # Leer frame de la webcam
        ret, frame = cap.read()
        
        # Convertir frame a RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
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
        cv2.imshow('Hand Tracking', cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
        
        # Guardar frame en el directorio correspondiente
        frame_path = os.path.join(letra_dir, f'{count}.jpg')
        cv2.imwrite(frame_path, frame_rgb)
        
        count += 1
        
        # Romper el bucle al presionar la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    

# Liberar recursos
cap.release()
cv2.destroyAllWindows()