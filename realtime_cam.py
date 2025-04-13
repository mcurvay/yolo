import cv2
from ultralytics import YOLO

# Modeli yükle
model = YOLO("runs/detect_quick/weights/best.pt")

# Kamerayı başlat (0 = dahili webcam)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Kamera açılamadı.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO tahmini (BGR yerine RGB veriyoruz)
    results = model.predict(source=frame, conf=0.5, verbose=False)

    # Sonuçları çizdir
    annotated_frame = results[0].plot()

    # Göster
    cv2.imshow("🍎 Elma Tanıma - YOLOv8", annotated_frame)

    # Çıkmak için ESC tuşuna bas
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Temizle
cap.release()
cv2.destroyAllWindows()
