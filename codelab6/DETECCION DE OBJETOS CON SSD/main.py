import ssl, certifi
import torch
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time

# 🔐 Forzar certificados válidos para evitar errores de SSL
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

# 📦 Cargar modelo SSD300-VGG16 preentrenado en COCO
weights = SSD300_VGG16_Weights.DEFAULT
model = ssd300_vgg16(weights=weights).eval()

# ⚙️ Transformaciones correctas del modelo (resize + normalización)
preprocess = weights.transforms()

# 📸 Cargar imagen y convertir a RGB
img = Image.open("perros.png").convert("RGB")

# 🧮 Preprocesar y añadir batch dimension
x = preprocess(img).unsqueeze(0)

# 🚀 Inferencia
with torch.no_grad():
    t0 = time.time()
    preds = model(x)[0]  # batch=1, tomamos la primera salida
    t1 = time.time()

print("Tiempo inferencia SSD:", round(t1 - t0, 3), "seg")

# 📊 Extraer predicciones
boxes, labels, scores = preds["boxes"], preds["labels"], preds["scores"]
categories = weights.meta["categories"]

# 🎨 Visualización
fig, ax = plt.subplots(1, figsize=(10, 8))
ax.imshow(img)

for box, lab, sc in zip(boxes, labels, scores):
    if sc > 0.5:  # umbral de confianza
        x1, y1, x2, y2 = box.tolist()
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor="red", facecolor="none"
        )
        ax.add_patch(rect)
        ax.text(
            x1, y1,
            f"{categories[int(lab)]}: {sc:.2f}",
            color="white", fontsize=8,
            bbox=dict(facecolor="black", alpha=0.5)
        )

plt.axis("off")
plt.show()
