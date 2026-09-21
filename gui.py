import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# 1. Mengatur tampilan halaman program
st.set_page_config(page_title="Identifikasi Penyakit Daun", layout="centered")
st.title("🌿 Program Identifikasi Penyakit Daun")
st.write("Unggah gambar daun untuk mendeteksi apakah daun tersebut sehat, terkena bacterial spot, atau early blight.")

# 2. Fungsi untuk memuat model (menggunakan cache agar program tidak lambat)
@st.cache_resource
def load_model():
    # Pastikan nama file model sesuai dengan yang Anda simpan sebelumnya
    return tf.keras.models.load_model('model_klasifikasi_daun.keras')

model = load_model()
class_names = ['bacterial_spot', 'early_blight', 'sehat']

# 3. Tombol untuk mengunggah gambar
uploaded_file = st.file_uploader("Pilih gambar daun...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 4. Tampilkan gambar yang diunggah
    image = Image.open(uploaded_file)
    st.image(image, caption='Gambar yang diuji', use_container_width=True)
    
    st.write("🔍 Sedang mengidentifikasi...")
    
    # 5. Proses gambar agar sesuai dengan input model (Resolusi 256x256)
    # Konversi gambar ke RGB untuk menghindari error jika gambar format PNG (RGBA)
    img_resized = image.convert('RGB').resize((256, 256)) 
    img_array = tf.keras.utils.img_to_array(img_resized)
    img_array = tf.expand_dims(img_array, 0) # Tambahkan dimensi batch
    
    # 6. Lakukan Prediksi
    prediksi = model.predict(img_array)
    indeks_hasil = np.argmax(prediksi[0])
    confidence = prediksi[0][indeks_hasil] * 100
    
    # 7. Tampilkan Hasil di bawah gambar
    st.success(f"**Hasil Identifikasi: {class_names[indeks_hasil].upper()}**")
    st.info(f"Tingkat Keyakinan (Confidence): {confidence:.2f}%")
