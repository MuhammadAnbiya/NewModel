import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# --- Konfigurasi dan Pemuatan Model ---
st.set_page_config(page_title="Deteksi Emosi Wajah", layout="centered")

@st.cache_resource
def load_emotion_model():
    model = load_model("emotion_model.h5", compile=False)
    return model

@st.cache_resource
def load_face_cascade():
    return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

emotion_model = load_emotion_model()
face_cascade = load_face_cascade()

ORIGINAL_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

def map_emotion(emotion):
    if emotion in ['Happy', 'Surprise']:
        return 'satisfied'
    elif emotion in ['Angry', 'Disgust', 'Fear', 'Sad']:
        return 'unsatisfied'
    else:
        return 'neutral'

# --- Fungsi untuk Memproses Gambar (Reusable) ---
def process_image(frame_bgr, show_messages=True):
    """Fungsi ini mengambil frame BGR dari OpenCV dan mengembalikan frame dengan deteksi."""
    # Salin frame agar frame asli tidak termodifikasi
    output_frame = frame_bgr.copy()
    
    gray = cv2.cvtColor(output_frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if show_messages and len(faces) == 0:
        st.warning("Tidak ada wajah yang terdeteksi.")
    
    if show_messages and len(faces) > 0:
        st.success(f"Terdeteksi {len(faces)} wajah!")

    for (x, y, w, h) in faces:
        cv2.rectangle(output_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi_gray = np.expand_dims(roi_gray, axis=0)
        roi_gray = np.expand_dims(roi_gray, axis=-1) / 255.0

        prediction = emotion_model.predict(roi_gray, verbose=0)[0]
        max_index = np.argmax(prediction)
        
        predicted_emotion = ORIGINAL_LABELS[max_index]
        final_emotion = map_emotion(predicted_emotion)
        
        label = f"{final_emotion}: {prediction[max_index]*100:.2f}%"
        cv2.putText(output_frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    return output_frame

# --- Antarmuka Pengguna (UI) Streamlit ---
st.title("🤖 Deteksi Emosi Wajah")
st.write("Aplikasi ini dapat mendeteksi emosi melalui unggahan gambar atau video webcam real-time.")

mode = st.sidebar.selectbox(
    "Pilih Mode:",
    ["🖼️ Unggah Gambar", "🎥 Deteksi Video dari Webcam"],
    key="mode_selector"
)

# Inisialisasi session state untuk kontrol loop
if 'run_webcam' not in st.session_state:
    st.session_state.run_webcam = False

if mode == "🎥 Deteksi Video dari Webcam":
    st.header("Deteksi Langsung dari Webcam")
    st.info("Klik 'Start' untuk memulai deteksi video dan 'Stop' untuk menghentikannya.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start"):
            st.session_state.run_webcam = True
    with col2:
        if st.button("Stop"):
            st.session_state.run_webcam = False
    
    # Buat placeholder untuk menampilkan frame video
    frame_placeholder = st.empty()
    
    if st.session_state.run_webcam:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Gagal membuka kamera.")
        else:
            while st.session_state.run_webcam:
                ret, frame = cap.read()
                if not ret:
                    st.error("Gagal membaca frame dari kamera. Menghentikan...")
                    st.session_state.run_webcam = False
                    break
                
                # Proses frame untuk deteksi
                processed_frame = process_image(frame, show_messages=False)
                
                # Konversi ke RGB untuk ditampilkan di Streamlit
                frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                
                # Tampilkan frame di placeholder
                frame_placeholder.image(frame_rgb, channels="RGB")
            
            # Lepaskan kamera setelah loop berhenti
            cap.release()

elif mode == "🖼️ Unggah Gambar":
    st.header("Analisis Emosi dari Gambar")
    uploaded_file = st.file_uploader("Pilih sebuah gambar...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        frame_rgb = np.array(image)
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        st.image(image, caption='Gambar yang Diunggah', use_column_width=True)
        processed_frame = process_image(frame_bgr, show_messages=True)
        processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        st.image(processed_frame_rgb, caption='Hasil Deteksi', use_column_width=True)