import os
import random
import shutil
import cv2
import numpy as np
import tensorflow as tf


train_dir = "C:/Users/syrin/OneDrive/Bureau/face_expression_recognition/train" #A ADAPTER : chemin avec notre "base de donnee" d'images

class_sizes = {cls: len(os.listdir(os.path.join(train_dir, cls))) for cls in os.listdir(train_dir)}
target_size = min(class_sizes.values())

print(f"Images par classe : {target_size}")

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest'
)

for emotion, count in class_sizes.items():
    path = os.path.join(train_dir, emotion)
    images = os.listdir(path)

   
    if count > target_size: #réduction
        print(f"Suppression {emotion} ({count} → {target_size})")
        images_to_remove = random.sample(images, count - target_size)
        for img in images_to_remove:
            os.remove(os.path.join(path, img))

   
    elif count < target_size: #augmentation
        print(f"Génération {emotion} ({count} → {target_size})")
        images = os.listdir(path)
        while len(os.listdir(path)) < target_size:
            img_name = random.choice(images)
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                continue

            img = cv2.resize(img, (48, 48))
            img = np.expand_dims(img, axis=-1)  # Ajouter une dimension pour Keras
            img = np.expand_dims(img, axis=0)   # Ajouter la batch dimension

            
            augmented_img = next(datagen.flow(img, batch_size=1))[0]

            
            new_img_name = f"{img_name.split('.')[0]}_aug{random.randint(1000, 9999)}.jpg" #on save
            new_img_path = os.path.join(path, new_img_name)
            cv2.imwrite(new_img_path, augmented_img.reshape(48, 48))

print("Dataset équilibré !")
