import os

train_dir = "C:/Users/syrin/OneDrive/Bureau/face_expression_recognition/train" #A ADAPTER


emotion_classes = os.listdir(train_dir) #on liste


for emotion in emotion_classes: #compte le nb d'images
    path = os.path.join(train_dir, emotion)
    num_images = len(os.listdir(path))
    print(f"{emotion}: {num_images} images")
