from ultralytics import YOLO

# 📦 Cargar modelo preentrenado
model_yolo = YOLO("yolov8n.pt")

# 🚀 Inferencia
results = model_yolo("perros.png")

# 👀 Mostrar resultado (toma el primero de la lista)
results[0].show()
