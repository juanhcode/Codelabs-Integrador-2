import cv2
from ultralytics import YOLO

# ------------------------------
# 1. Cargar el modelo YOLOv8
# ------------------------------
model = YOLO("yolov8n.pt")  # modelo liviano

# ------------------------------
# 2. Intentar abrir la cámara
# ------------------------------
cap = cv2.VideoCapture(0)  # usa 1 si la cámara integrada no responde

if not cap.isOpened():
    print("⚠️ No se pudo acceder a la cámara. Revisa permisos en macOS.")
    exit()

# Opcional: forzar resolución
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ------------------------------
# 3. Loop de captura y detección
# ------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ No se pudo leer el frame de la cámara.")
        break

    # Ejecutar YOLO en el frame
    results = model(frame)

    # Dibujar las detecciones en la imagen
    annotated = results[0].plot()

    # Mostrar ventana
    cv2.imshow("Detección YOLOv8 - Cámara", annotated)

    # Presiona "q" para salir
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ------------------------------
# 4. Liberar recursos
# ------------------------------
cap.release()
cv2.destroyAllWindows()
