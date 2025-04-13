from ultralytics import YOLO

# Modeli yükle (hafif model)
model = YOLO("yolov8n.pt")

# Eğitimi başlat
model.train(
    data="/Users/cagatay/Documents/comp.eng/comp.vision/yolo/dataset.yaml",
    epochs=20,              # Eğitim süresi kısaldı
    imgsz=416,              # Görüntü boyutu daha küçük → hız artışı
    batch=32,               # Daha büyük batch → daha hızlı convergence
    project="runs",
    name="detect_quick",
    exist_ok=True
)

print("✅ Eğitim tamamlandı: runs/detect/detect_quick klasörünü kontrol et.")
