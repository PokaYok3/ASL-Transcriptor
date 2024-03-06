import cv2
import mediapipe as mp
import os
import time
# Crear directorio para almacenar el dataset
dataset_dir = './dataset2/train'
if not os.path.exists(dataset_dir):
    print("Creando directorio para almacenar el dataset")
    os.makedirs(dataset_dir)

# Inicializar Mediapipe Hand module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands()

# Inicializar webcam
cap = cv2.VideoCapture(0)

# Definir las letras del abecedario
letras = 'n'


# Capturar 30 frames de cada letra
for letra in letras:
    count=  0
    if (letra=='1'):
        letra="space"
    print("Letra: ", letra)
    
    letra_dir = os.path.join(dataset_dir, letra)
    if not os.path.exists(letra_dir):
        os.makedirs(letra_dir)
    print("Se va a capturar la letra ", letra)
    time.sleep(10)

    print("Capturando  frames de la letra ", letra)

    start_time = time.time()
    while time.time() - start_time < 12:
        
        data_aux = []
        x_ = []
        y_ = []
        # Leer frame de la webcam
        ret, frame = cap.read()
        H, W, _ = frame.shape
        
        # Convertir frame a RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Procesar frame con Mediapipe Hand module
        results = hands.process(frame_rgb)
    
        if results.multi_hand_landmarks:
            # Asignar landmarks de la mano
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,  # image to draw
                    hand_landmarks,  # model output
                    mp_hands.HAND_CONNECTIONS,  # hand connections
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)
            x1 = int(min(x_) * W) - 50
            y1 = int(min(y_) * H) - 20

            x2 = int(max(x_) * W) + 30
            y2 = int(max(y_) * H) + 30
            # Create a new window to display the frame within the rectangle
            width = 64  # desired width of the resized image
            height = 64  # desired height of the resized image
            frame_with_rectangle = cv2.resize(frame[y1:y2, x1:x2], (128, 128))
            cv2.imshow('Frame with Rectangle', frame_with_rectangle)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)

            
            # Mostrar frame
            cv2.imshow('frame', frame)        
            # Guardar frame en el directorio correspondiente
            frame_path = os.path.join(letra_dir, f'{count}.jpg')
            cv2.imwrite(frame_path, frame_with_rectangle)
        
        count += 1
        
        # Romper el bucle al presionar la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    time.sleep(5)

# Liberar recursos
cap.release()
cv2.destroyAllWindows()