import cv2, dlib

# ----- Inicializar cámara -----
cap = cv2.VideoCapture(0)  # 0 = cámara por defecto

if not cap.isOpened():
    print("⚠️ No se pudo abrir la cámara. Intentando con archivo de video...")
    cap = cv2.VideoCapture("video.mp4")  # Usa un video si falla la cámara

# Detector HOG de dlib
detector = dlib.get_frontal_face_detector()

print("✅ Presiona ESC para salir")

frame_count = 0
rects = []  # almacenar rostros detectados

while True:
    ok, frame = cap.read()
    if not ok:
        print("⚠️ No se pudo leer frame. Saliendo...")
        break

    frame_count += 1

    # Solo detectar cada 2 frames (para reducir lag)
    if frame_count % 2 == 0:
        # Reducir resolución para acelerar
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

        # Detección de rostros
        rects = detector(gray, 0)

        # Escalar coordenadas al tamaño original
        scaled_rects = []
        for r in rects:
            x, y, w, h = r.left()*2, r.top()*2, r.width()*2, r.height()*2
            scaled_rects.append((x, y, w, h))
        rects = scaled_rects

    # Dibujar rectángulos (se reutilizan hasta la siguiente detección)
    for (x, y, w, h) in rects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Mostrar ventana en colores normales
    cv2.imshow("dlib HOG (Optimizado)", frame)

    # Salida con ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

# ----- Liberar recursos -----
cap.release()
cv2.destroyAllWindows()
