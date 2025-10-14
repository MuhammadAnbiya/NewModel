import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase # DIUBAH
import av # DIUBAH: Impor library av

# --- Konfigurasi dan Pemuatan Model ---
st.set_page_config(page_title="Deteksi Emosi Wajah", layout="centered")

@st.cache_resource
def load_emotion_model():
    model = load_model("emotion_model.h5", compile=False) # Tambahkan compile=False untuk stabilitas
    return model

emotion_model = load_emotion_model()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

ORIGINAL_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

def map_emotion(emotion):
    if emotion in ['Happy', 'Surprise']:
        return 'satisfied'
    elif emotion in ['Angry', 'Disgust', 'Fear', 'Sad']:
        return 'unsatisfied'
    else:
        return 'neutral'

# --- DIUBAH: Kelas Proses Frame Menggunakan API BARU ---
class EmotionProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_count = 0
        self.last_label = "Mendeteksi..."

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        if self.frame_count % 5 == 0:
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y + h, x:x + w]
                roi_gray = cv2.resize(roi_gray, (48, 48))
                roi_gray = np.expand_dims(roi_gray, axis=0)
                roi_gray = np.expand_dims(roi_gray, axis=-1) / 255.0

                prediction = emotion_model.predict(roi_gray, verbose=0)[0]
                max_index = np.argmax(prediction)
                
                predicted_emotion = ORIGINAL_LABELS[max_index]
                final_emotion = map_emotion(predicted_emotion)
                
                self.last_label = f"{final_emotion}: {prediction[max_index]*100:.2f}%"
        
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(img, self.last_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        
        self.frame_count += 1
        
        # Kembalikan frame sebagai av.VideoFrame
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- Antarmuka Pengguna (UI) Streamlit ---
st.title("🤖 Deteksi Emosi Wajah")
st.write("Aplikasi ini dapat mendeteksi emosi 'satisfied', 'unsatisfied', atau 'neutral' dari wajah secara real-time atau melalui unggahan gambar.")

mode = st.sidebar.selectbox("Pilih Mode:", ["🎥 Deteksi via Webcam", "🖼️ Unggah Gambar"])

if mode == "🎥 Deteksi via Webcam":
    st.header("Deteksi Langsung dari Webcam")
    st.info("Klik 'START' untuk memulai deteksi. Berikan izin akses kamera jika diminta.")
    # DIUBAH: Gunakan video_processor_factory dan EmotionProcessor
    webrtc_streamer(
        key="webcam", 
        video_processor_factory=EmotionProcessor,
        rtc_configuration={ # Tambahkan konfigurasi ini untuk koneksi yang lebih stabil
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        }
    )

elif mode == "🖼️ Unggah Gambar":
    st.header("Analisis Emosi dari Gambar")
    # ... (Bagian unggah gambar tidak perlu diubah, sudah benar) ...
    uploaded_file = st.file_uploader("Pilih sebuah gambar...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Gambar yang Diunggah', use_column_width=True)
        
        frame = np.array(image)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        if len(faces) == 0:
            st.warning("Tidak ada wajah yang terdeteksi pada gambar.")
        else:
            st.success(f"{len(faces)} wajah terdeteksi!")
            for (x, y, w, h) in faces:
                cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (255, 0, 0), 2)
                roi_gray = gray[y:y + h, x:x + w]
                roi_gray = cv2.resize(roi_gray, (48, 48))
                roi_gray = np.expand_dims(roi_gray, axis=0)
                roi_gray = np.expand_dims(roi_gray, axis=-1) / 255.0

                prediction = emotion_model.predict(roi_gray, verbose=0)[0]
                max_index = np.argmax(prediction)
                
                predicted_emotion = ORIGINAL_LABELS[max_index]
                final_emotion = map_emotion(predicted_emotion)
                
                label = f"{final_emotion}: {prediction[max_index]*100:.2f}%"
                cv2.putText(frame_bgr, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            
            result_image = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            st.image(result_image, caption='Hasil Deteksi', use_column_width=True)