import streamlit as st
from PIL import Image
from ultralytics import YOLO
import os

# Modeli yükle
model = YOLO("runs/detect_quick/weights/best.pt")

# Başlık
st.title("🍎 Elma Tanıma Uygulaması (YOLOv8)")
st.write("Görsel yükleyin, modelimiz elma olup olmadığını tahmin etsin!")

# Görsel yükleme
uploaded_file = st.file_uploader("Görsel seç (.jpg/.png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Görseli aç
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Yüklenen Görsel", use_column_width=True)

    # Görseli geçici olarak kaydet
    temp_path = "temp_uploaded_image.jpg"
    image.save(temp_path)

    # Model ile tahmin yap
    results = model.predict(source=temp_path, save=True, conf=0.5)

    # Sonuç görselini yükle ve göster
    result_path = os.path.join(results[0].save_dir, os.path.basename(results[0].path))

    st.image(result_path, caption="Tahmin Sonucu", use_column_width=True)

    # Geçici dosyayı sil
    os.remove(temp_path)
