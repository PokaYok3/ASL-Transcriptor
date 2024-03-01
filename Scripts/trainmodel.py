import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from datetime import datetime
import os
import matplotlib.pyplot as plt

#Create data generator for training
def create_data_train_generator(train_path):
    train_datagen = ImageDataGenerator(rescale=1./255,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True)

    train_generator = train_datagen.flow_from_directory(train_path,target_size=(64, 64),batch_size=32,class_mode='categorical')

   
    return train_generator

#Create CNN model
def create_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(27, activation='softmax'))
    return model

train_generator=create_data_train_generator('./dataset/train')
model=create_model()
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])




model.fit(train_generator, epochs=50)

actualtime = datetime.now()

#Formatear la hora
timestamp = actualtime.strftime("%Y%m%d_%H%M%S")
nombre_archivo = 'prueba2.keras'
# Guardar el modelo 
model.save(nombre_archivo)
print(f"Modelo guardado en {nombre_archivo}")



