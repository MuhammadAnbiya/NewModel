import tensorflow as tf
from tensorflow.keras import layers, models
from load_data import train_generator, test_generator

#modèle CNN
model = models.Sequential([
    layers.Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(7, activation='softmax')  # 7 = émotions
])

#compilation
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


model.fit(train_generator, epochs=10, validation_data=test_generator)#entrainement


model.save('emotion_model.h5') #sauvegarde
print("Modèle entraîné et sauvegardé sous emotion_model.h5")
