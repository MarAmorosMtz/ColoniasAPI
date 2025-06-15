import cv2
import numpy as np


def procesar_subcolonias(imagen_original, contorno, tipo_colonia, numero_colonia):
    # Crear máscara para la colonia actual
    mask = np.zeros_like(imagen_original[:, :, 0])
    cv2.drawContours(mask, [contorno], -1, 255, -1)

    # ROI (Region of Interest)
    x, y, w, h = cv2.boundingRect(contorno)
    roi = imagen_original[y:y + h, x:x + w]
    roi_mask = mask[y:y + h, x:x + w]

    # Aplicar watershed solo en la ROI
    sure_bg = cv2.dilate(roi_mask, np.ones((3, 3), np.uint8), iterations=3)
    dist_transform = cv2.distanceTransform(roi_mask, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    markers = cv2.watershed(roi, markers)
    roi[markers == -1] = [0, 255, 0]  # Bordes verdes

    # Contar subcolonias (marcadores únicos)
    subcolonias = len(np.unique(markers)) - 2  # Restamos fondo y bordes

    # Dibujar número de colonia
    center = (int(x + w / 2), int(y + h / 2))
    cv2.putText(imagen_original, str(numero_colonia), center,
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return subcolonias


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
lower_red1 = np.array([0, 70, 50])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 70, 50])
upper_red2 = np.array([180, 255, 255])
lower_blue = np.array([90, 50, 50])
upper_blue = np.array([130, 255, 255])
lower_brown = np.array([10, 60, 40])
upper_brown = np.array([20, 200, 150])

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
mask_brown = cv2.morphologyEx(mask_brown, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2)

# Encontrar contornos principales
contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours_brown, _ = cv2.findContours(mask_brown, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Procesamiento principal
min_area = 50
count_red = count_blue = count_brown = 0
total_colonias = 0

# Procesar colonias ROJAS
for i, cnt in enumerate(contours_red, start=1):
    area = cv2.contourArea(cnt)
    if area > min_area:
        subcolonias = procesar_subcolonias(img_original, cnt, "R", i)
        count_red += max(1, subcolonias)
        total_colonias += max(1, subcolonias)

# Procesar colonias AZULES
for i, cnt in enumerate(contours_blue, start=len(contours_red) + 1):
    area = cv2.contourArea(cnt)
    if area > min_area:
        subcolonias = procesar_subcolonias(img_original, cnt, "A", i)
        count_blue += max(1, subcolonias)
        total_colonias += max(1, subcolonias)

# Procesar colonias CAFÉ
for i, cnt in enumerate(contours_brown, start=len(contours_red) + len(contours_blue) + 1):
    area = cv2.contourArea(cnt)
    if area > min_area:
        subcolonias = procesar_subcolonias(img_original, cnt, "C", i)
        count_brown += max(1, subcolonias)
        total_colonias += max(1, subcolonias)

# Mostrar resumen en la imagen
resumen_texto = f"Total: {total_colonias} (R:{count_red} A:{count_blue} C:{count_brown})"
cv2.putText(img_original, resumen_texto, (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

# Resultados en consola
print(f"🔴 Colonias rojas: {count_red}")
print(f"🔵 Colonias azules: {count_blue}")
print(f"🟤 Colonias café: {count_brown}")
print(f"🚀 Total de colonias: {total_colonias}")

cv2.imwrite('resultado_final.jpg', img_original)
