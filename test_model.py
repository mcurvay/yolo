from ultralytics import YOLO
from PIL import Image

# Modeli yükle
model = YOLO("runs/detect_quick/weights/best.pt")

# Test etmek istediğin görseli buraya yaz
image_path = "images/val/elma1.jpg"  # örnek bir görsel

# Tahmin yap
results = model.predict(source=image_path, save=True, conf=0.5)

# Sonucu göster
Image.open(results[0].save_path).show()
