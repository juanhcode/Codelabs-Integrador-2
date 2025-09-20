# Comparación SSD vs YOLOv8 🚀

Este proyecto compara el rendimiento de dos modelos de detección de objetos: **SSD** y **YOLOv8**, utilizando tres imágenes de prueba (`perros.png`, `persona.jpg`, `personas.jpg`).

## 📊 Resultados comparativos

| Imagen        | Tiempo SSD (s) | Tiempo YOLO (s) | Objetos SSD | Objetos YOLO | IoU SSD-YOLO |
|---------------|----------------|-----------------|-------------|--------------|--------------|
| perros.png    | 0.246          | 0.348           | 6           | 6            | N/A          |
| persona.jpg   | 0.161          | 0.065           | 1           | 1            | 0.901        |
| personas.jpg  | 0.174          | 0.067           | 5           | 5            | 0.901        |

## ✅ Conclusión

- Ambos modelos detectan correctamente los objetos en las imágenes.  
- **SSD** muestra tiempos de inferencia más altos en la mayoría de los casos (0.16–0.24 s).  
- **YOLOv8** es más rápido en las imágenes de `persona.jpg` y `personas.jpg`, con tiempos cercanos a 0.06–0.07 s, lo que lo hace más eficiente para escenarios donde la latencia es crítica.  
- El **IoU (Intersection over Union)** entre SSD y YOLO es alto (≈0.90), lo que indica que ambos modelos hacen predicciones muy similares.  

👉 **Para un proyecto en tiempo real, escogería YOLOv8**, ya que ofrece **mejor velocidad sin sacrificar precisión**, lo cual es clave en aplicaciones como videovigilancia, conducción autónoma o cualquier tarea que requiera detecciones inmediatas.
