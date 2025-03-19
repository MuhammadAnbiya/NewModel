import tensorflow as tf
import os
import cv2
import numpy as np


train_dir = "C:/Users/syrin/OneDrive/Bureau/face_expression_recognition/train" # A ADAPTER : chemin avec notre "base de donnée" d'image 


target_classes = ["disgust", "fear", "sad", "surprise"]  #le fer2013 possède des classes sous représentés donc on les adapte à notre modèle


datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest'
)


for emotion in target_classes: #création d'image
    path = os.path.join(train_dir, emotion)
    images = os.listdir(path)

    for img_name in images:
        img_path = os.path.join(path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"Erreur, {img_path}")
            continue

        img = cv2.resize(img, (48, 48))
        img = np.expand_dims(img, axis=-1) 
        img = np.expand_dims(img, axis=0)   

        
        for i in range(3):
            augmented_img = next(datagen.flow(img, batch_size=1))  # ✅ Correction ici
            new_img_name = f"{img_name.split('.')[0]}_aug{i}.jpg"
            new_img_path = os.path.join(path, new_img_name)
            cv2.imwrite(new_img_path, augmented_img[0].reshape(48, 48))

    print(f"Augmentation effectuée {emotion}. Nouvelle taille : {len(os.listdir(path))} images")

print("Toutes les augmentations sont terminées !")