import cv2
import numpy as np
import matplotlib.pyplot as plt
from mtcnn.mtcnn import MTCNN
from time import time
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

# ---------- CONFIG ----------
imagenes = ["persona.jpg", "grupo.jpg", "parciales.jpg"]  # 3 imágenes a evaluar
umbrales = [0.60, 0.75, 0.90, 0.95]  # umbrales de confianza
runs = 5  # número de corridas para promediar tiempos
detector = MTCNN()

# ---------- FUNCIONES ----------
def detectar(img_path, thr=0.90):
    """Detecta rostros en una imagen, filtra por umbral y devuelve métricas"""
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Medir tiempo promedio
    tiempos = []
    for _ in range(runs):
        t0 = time()
        res = detector.detect_faces(img_rgb)
        t1 = time()
        tiempos.append(t1 - t0)

    # Filtrar por confianza
    filtrados = [r for r in res if r['confidence'] >= thr]

    # Dibujar detecciones
    vis = img_rgb.copy()
    for r in filtrados:
        x, y, w, h = r['box']
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Landmarks
        for name, (px, py) in r['keypoints'].items():
            cv2.circle(vis, (px, py), 3, (255, 0, 0), -1)
            cv2.putText(vis, name, (px + 2, py - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # Blur opcional para privacidad
        face = vis[y:y+h, x:x+w]
        if face.size > 0:
            face_blur = cv2.GaussianBlur(face, (51, 51), 30)
            vis[y:y+h, x:x+w] = face_blur

    # Mostrar imagen procesada
    plt.imshow(vis)
    plt.title(f"{img_path} | thr={thr}")
    plt.axis("off")
    plt.show()

    return {
        "imagen": img_path,
        "umbral": thr,
        "rostros_detectados": len(res),
        "rostros_filtrados": len(filtrados),
        "tiempo_prom_ms": np.mean(tiempos) * 1000
    }

# ---------- EXPERIMENTO ----------
resultados = []
for img in imagenes:
    for thr in umbrales:
        res = detectar(img, thr=thr)
        resultados.append(res)

# ---------- REPORTE ----------
# Convertir resultados a DataFrame
df = pd.DataFrame(resultados)
print(df)

# Crear PDF
pdf_filename = "reporte_resultados.pdf"
doc = SimpleDocTemplate(pdf_filename, pagesize=letter)
elements = []

# Título
styles = getSampleStyleSheet()
title = Paragraph("Reporte de Detección de Rostros", styles["Title"])
elements.append(title)
elements.append(Spacer(1, 12))

# Convertir DataFrame a tabla
data = [df.columns.tolist()] + df.values.tolist()
table = Table(data)

# Estilos de la tabla
style = TableStyle([
    ('BACKGROUND', (0,0), (-1,0), colors.grey),
    ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
    ('ALIGN', (0,0), (-1,-1), 'CENTER'),
    ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
    ('BOTTOMPADDING', (0,0), (-1,0), 12),
    ('BACKGROUND', (0,1), (-1,-1), colors.beige),
    ('GRID', (0,0), (-1,-1), 1, colors.black),
])
table.setStyle(style)

elements.append(table)

# Construir PDF
doc.build(elements)
print(f"✅ PDF generado: {pdf_filename}")
