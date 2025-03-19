import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


train_dir = 'C:/Users/syrin/OneDrive/Bureau/face_expression_recognition/train' #A ADAPTER
test_dir = 'C:/Users/syrin/OneDrive/Bureau/face_expression_recognition/test' #A ADAPTER


train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),  # Redimensionner les images à 48x48
    color_mode='grayscale',  # Charger les images en niveaux de gris
    batch_size=32,
    class_mode='categorical')  # Categorical pour avoir des labels one-hot


test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(48, 48),  #on redimensionne 
    color_mode='grayscale',  #ton grisatre
    batch_size=32,
    class_mode='categorical') 


__all__ = ['train_generator', 'test_generator'] #accessibilité
