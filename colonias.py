import cv2
import numpy as np

# Cargar imagen
img = cv2.imread('HCL001R.jpg')
img_original = img.copy()

# 1. Preprocesamiento
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)
clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16, 16))
l = clahe.apply(l)
lab = cv2.merge((l, a, b))
img_enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

# Convertir a HSV para detección de color
hsv = cv2.cvtColor(img_enhanced, cv2.COLOR_BGR2HSV)

# Rangos de color ajustados
## Rojo
lower_red1 = np.array([0, 70, 50])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 70, 50])
upper_red2 = np.array([180, 255, 255])

## Azul
lower_blue = np.array([90, 50, 50])
upper_blue = np.array([130, 255, 255])

# Rangos de color ajustados para café fuerte (nuevos valores)
lower_brown = np.array([10, 60, 40])    # H:10-20, S más alto (60), V moderado (40)
upper_brown = np.array([20, 200, 150])  # V máximo reducido para evitar claros

# Máscaras para cada color
mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
mask_red = cv2.bitwise_or(mask_red1, mask_red2)
mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)

# Mejorar máscaras con morfología
kernel = np.ones((5, 5), np.uint8)
mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel, iterations=2)
mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel, iterations=2)
mask_brown = cv2.morphologyEx(mask_brown, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8), iterations=2)

# Encontrar contornos
contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours_brown, _ = cv2.findContours(mask_brown, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Contornos café

# Filtrar por área y dibujar resultados
min_area = 20
count_red = 0
count_blue = 0
count_brown = 0  # Contador para colonias café

# Procesar colonias rojas
for cnt in contours_red:
    area = cv2.contourArea(cnt)
    if area > min_area:
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        radius = int(radius)
        cv2.circle(img_original, center, radius, (0, 255, 0), 2)
        cv2.putText(img_original, "R", (center[0], center[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        count_red += 1

# Procesar colonias azules
for cnt in contours_blue:
    area = cv2.contourArea(cnt)
    if area > min_area:
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        radius = int(radius)
        cv2.circle(img_original, center, radius, (0, 255, 0), 2)
        cv2.putText(img_original, "A", (center[0], center[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        count_blue += 1

# Procesar colonias café-amarillentas (NUEVO)
for cnt in contours_brown:
    area = cv2.contourArea(cnt)
    if area > min_area:
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        radius = int(radius)
        cv2.circle(img_original, center, radius, (0, 255, 0), 2)
        cv2.putText(img_original, "C", (center[0], center[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (42, 42, 165), 2)  # Color café
        count_brown += 1

# Resultados
print(f"🔴 Colonias rojas: {count_red}")
print(f"🔵 Colonias azules: {count_blue}")
print(f"🟤 Colonias café-amarillentas: {count_brown}")  # Nuevo resultado

# Mostrar y guardar
cv2.imwrite('resultado_final.jpg', img_original)