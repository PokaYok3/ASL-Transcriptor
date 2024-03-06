import pickle
import time
import cv2
import mediapipe as mp
import numpy as np
import os
from tensorflow.keras.models import load_model
import pyttsx3
import os
# Carga el modelo
model_path = "./prueba4.keras"
model = load_model(model_path)
cap = cv2.VideoCapture(0)
def convert_predict_label(predicted_class):
    labels_dict = {}
    train_dir = './dataset2/train'
    i=0
    for label in os.listdir(train_dir):

        if os.path.isdir(os.path.join(train_dir, label)):
            labels_dict[int(i)] = label
        i+=1
    return labels_dict[predicted_class]
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
letras_string = ""
while cv2.waitKey(1) != 13:  # 13 is the ASCII code for the Enter key
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
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

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        x1 = int(min(x_) * W) - 50
        y1 = int(min(y_) * H) - 20

        x2 = int(max(x_) * W) + 30
        y2 = int(max(y_) * H) + 30


        
        # Create a new window to display the frame within the rectangle
        width = 64  # desired width of the resized image
        height = 64  # desired height of the resized image
        frame_with_rectangle = cv2.resize(frame[y1:y2, x1:x2], (64, 64))
        img = frame_with_rectangle / 255.0
        img=np.expand_dims(img, axis=0)
        prediction=model.predict(img)
        predicted_class = np.argmax(prediction)
        print("Prediction", predicted_class)
        letra_=convert_predict_label(predicted_class)
        print("Letra: ", letra_)
        letras_string += letra_ + " " if letra_ != "space" else " "
        cv2.imshow('Frame with Rectangle', frame_with_rectangle)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)

    time.sleep(5)
    cv2.imshow('frame', frame)


cap.release()
cv2.destroyAllWindows()
# Transcribe the phrase to audio

engine = pyttsx3.init()
engine.save_to_file(letras_string, 'transcription.mp3')
engine.runAndWait()

# Play the audio
os.system('start transcription.mp3')
