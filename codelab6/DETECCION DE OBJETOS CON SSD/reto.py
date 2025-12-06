import ssl, certifi, time
import torch
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
from PIL import Image
from ultralytics import YOLO
import pandas as pd

# =========================
# 🔹 Función IoU
# =========================
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

# =========================
# 🔹 Modelos
# =========================
# YOLOv8
model_yolo = YOLO("yolov8n.pt")

# SSD
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())
weights = SSD300_VGG16_Weights.DEFAULT
model_ssd = ssd300_vgg16(weights=weights).eval()
preprocess = weights.transforms()

# =========================
# 🔹 Evaluar imágenes
# =========================
imagenes = ["perros.png", "persona.jpg", "personas.jpg"]
resultados = []

for img_path in imagenes:
    print(f"\n📸 Procesando {img_path}...")

    # ----- YOLO -----
    t0 = time.time()
    results = model_yolo(img_path)
    t1 = time.time()
    yolo_time = t1 - t0
    boxes_yolo = results[0].boxes.xyxy.cpu().numpy()
    scores_yolo = results[0].boxes.conf.cpu().numpy()
    num_yolo = (scores_yolo > 0.5).sum()
    box_yolo = boxes_yolo[scores_yolo.argmax()] if len(scores_yolo) > 0 else None

    # ----- SSD -----
    img = Image.open(img_path).convert("RGB")
    x = preprocess(img).unsqueeze(0)
    t0 = time.time()
    with torch.no_grad():
        preds = model_ssd(x)[0]
    t1 = time.time()
    ssd_time = t1 - t0
    boxes_ssd = preds["boxes"].cpu().numpy()
    scores_ssd = preds["scores"].cpu().numpy()
    num_ssd = (scores_ssd > 0.5).sum()
    box_ssd = boxes_ssd[scores_ssd.argmax()] if len(scores_ssd) > 0 else None

    # ----- IoU -----
    if box_yolo is not None and box_ssd is not None:
        iou_val = iou(box_yolo, box_ssd)
    else:
        iou_val = None

    resultados.append({
        "Imagen": img_path,
        "Tiempo SSD (s)": round(ssd_time, 3),
        "Tiempo YOLO (s)": round(yolo_time, 3),
        "Objetos SSD": int(num_ssd),
        "Objetos YOLO": int(num_yolo),
        "IoU SSD-YOLO": round(iou_val, 3) if iou_val else "N/A"
    })

# =========================
# 🔹 Mostrar tabla
# =========================
df = pd.DataFrame(resultados)
print("\n📊 Resultados comparativos:")
print(df.to_string(index=False))
