import ssl, certifi
import torch
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
from PIL import Image
from ultralytics import YOLO

# =========================
# 🔹 Definir función IoU
# =========================
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return interArea / float(boxAArea + boxBArea - interArea)

# =========================
# 🔹 YOLOv8 detección
# =========================
model_yolo = YOLO("yolov8n.pt")
results = model_yolo("perros.png")
boxes_yolo = results[0].boxes.xyxy.cpu().numpy()   # (N, 4)
scores_yolo = results[0].boxes.conf.cpu().numpy()

# Tomamos la caja con mayor score de YOLO
idx_yolo = scores_yolo.argmax()
box_yolo = boxes_yolo[idx_yolo]
print("Mejor caja YOLO:", box_yolo, "Conf:", scores_yolo[idx_yolo])

# =========================
# 🔹 SSD detección
# =========================
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())
weights = SSD300_VGG16_Weights.DEFAULT
model_ssd = ssd300_vgg16(weights=weights).eval()

img = Image.open("perros.png").convert("RGB")
preprocess = weights.transforms()
x = preprocess(img).unsqueeze(0)

with torch.no_grad():
    preds = model_ssd(x)[0]

boxes_ssd = preds["boxes"].cpu().numpy()
scores_ssd = preds["scores"].cpu().numpy()

# Tomamos la caja con mayor score de SSD
idx_ssd = scores_ssd.argmax()
box_ssd = boxes_ssd[idx_ssd]
print("Mejor caja SSD:", box_ssd, "Conf:", scores_ssd[idx_ssd])

# =========================
# 🔹 Calcular IoU
# =========================
iou_value = iou(box_yolo, box_ssd)
print(f"IoU entre mejor caja YOLO y mejor caja SSD: {iou_value:.3f}")
