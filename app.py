# import cv2
# import numpy as np
# from tensorflow.keras.models import load_model


# emotion_model = load_model("emotion_model.h5") #on charge notre modèle


# emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'] #taxinomie des emotions


# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') # on l'initialise


# cap = cv2.VideoCapture(0) 

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

    
#     frame = cv2.flip(frame, 1) #effet inversé de la cam (rendement plus esthétique)

    
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #ton gris
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

#     for (x, y, w, h) in faces:
#         roi_gray = gray[y:y+h, x:x+w]
#         roi_gray = cv2.resize(roi_gray, (48, 48))
#         roi_gray = np.expand_dims(roi_gray, axis=0)
#         roi_gray = np.expand_dims(roi_gray, axis=-1) / 255.0

        
#         predictions = emotion_model.predict(roi_gray)[0]
#         max_index = np.argmax(predictions)
#         emotion = emotion_labels[max_index]

        
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2) #cadre sur le visage 
#         cv2.putText(frame, f"{emotion}: {round(predictions[max_index]*100, 2)}%", #écriture correspondant aux émotions
#                     (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    
#     cv2.imshow("Emotion Detection", frame)

    
#     if cv2.waitKey(1) & 0xFF == ord('q'): #on quitte la vidéo en appuyant sur la touche "q"
#         break


# cap.release()
# cv2.destroyAllWindows()


import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Muat model yang sudah ada (yang dilatih untuk 7 emosi)
emotion_model = load_model("emotion_model.h5") 

# Label asli dari model
original_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Fungsi untuk memetakan 7 emosi ke 3 kategori
def map_emotion(emotion):
    if emotion == 'Happy':
        return 'satisfied'
    elif emotion in ['Angry', 'Disgust', 'Fear', 'Sad']:
        return 'unsatisfied'
    else:  # Neutral, Surprise
        return 'netral'

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') 
cap = cv2.VideoCapture(0) 

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi_gray = np.expand_dims(roi_gray, axis=0)
        roi_gray = np.expand_dims(roi_gray, axis=-1) / 255.0

        # Model tetap memprediksi dari 7 kelas
        predictions = emotion_model.predict(roi_gray)[0]
        max_index = np.argmax(predictions)
        
        # Dapatkan nama emosi asli
        predicted_emotion = original_labels[max_index]
        
        # Terjemahkan ke 3 kategori baru
        final_emotion = map_emotion(predicted_emotion)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # Tampilkan hasil terjemahan
        cv2.putText(frame, f"{final_emotion}: {round(predictions[max_index]*100, 2)}%",
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    cv2.imshow("Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()