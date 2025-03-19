import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt


model = tf.keras.models.load_model('emotion_model.h5') #chargement


img_path = 'C:/Users/syrin/OneDrive/Bureau/face_espression_recognition/test/happy/PrivateTest_95094.jpg'  #A ADAPTER : si on veut tester une image de notre dossier pour vérifier qu'il est compétant


img = image.load_img(img_path, target_size=(48, 48), color_mode='grayscale')
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  
img_array = img_array / 255.0  # Normalisation


prediction = model.predict(img_array)
predicted_class = np.argmax(prediction)  #prédiction


class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
print(f"Emotion prédite : {class_names[predicted_class]}")


plt.imshow(img, cmap="gray") #affichage
plt.title(f"Emotion: {class_names[predicted_class]}")
plt.show()
