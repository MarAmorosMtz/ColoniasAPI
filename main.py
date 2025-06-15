import io
import os
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.responses import JSONResponse, FileResponse
from typing import Dict, List
from pydantic import BaseModel
from enum import Enum

app = FastAPI(title="API de Análisis de Colonias Bacterianas",
              description="Procesa imágenes JPG de placas de agar usando el algoritmo específico")

# Configuración de directorios
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "output"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# Modelos de datos
class ColonyType(str, Enum):
    RED = "red"
    BLUE = "blue"
    BROWN = "brown"


class ColonyData(BaseModel):
    id: int
    type: ColonyType
    position: Dict[str, int]
    radius: int
    area: float
    subcolonies: int


class AnalysisResult(BaseModel):
    total_colonies: int
    red_colonies: int
    blue_colonies: int
    brown_colonies: int
    colonies: List[ColonyData]


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


def procesar_imagen_jpg(file_contents: bytes) -> tuple:
    """Procesa una imagen JPG usando el algoritmo específico"""
    try:
        # Convertir bytes a imagen OpenCV
        nparr = np.frombuffer(file_contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("No se pudo decodificar la imagen JPG")

        img_original = img.copy()
        height, width = img.shape[:2]

        # 1. Preprocesamiento (igual que en tu código)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16, 16))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        img_enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # 2. Detección de color (HSV)
        hsv = cv2.cvtColor(img_enhanced, cv2.COLOR_BGR2HSV)

        # Rangos de color (igual que en tu código)
        lower_red1 = np.array([0, 70, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 70, 50])
        upper_red2 = np.array([180, 255, 255])
        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([130, 255, 255])
        lower_brown = np.array([10, 60, 40])
        upper_brown = np.array([20, 200, 150])

        # Máscaras de color (igual que en tu código)
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)

        # Operaciones morfológicas (igual que en tu código)
        kernel = np.ones((5, 5), np.uint8)
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask_brown = cv2.morphologyEx(mask_brown, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2)

        # Encontrar contornos principales (igual que en tu código)
        contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_brown, _ = cv2.findContours(mask_brown, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Procesamiento principal (igual que en tu código)
        min_area = 50
        count_red = count_blue = count_brown = 0
        total_colonias = 0
        colonies_data = []

        # Procesar colonias ROJAS
        for i, cnt in enumerate(contours_red, start=1):
            area = cv2.contourArea(cnt)
            if area > min_area:
                subcolonias = procesar_subcolonias(img_original, cnt, ColonyType.RED, i)
                count_red += max(1, subcolonias)
                total_colonias += max(1, subcolonias)

                (x, y), radius = cv2.minEnclosingCircle(cnt)
                colonies_data.append({
                    "id": i,
                    "type": ColonyType.RED,
                    "position": {"x": int(x), "y": int(y)},
                    "radius": int(radius),
                    "area": float(area),
                    "subcolonies": subcolonias
                })

        # Procesar colonias AZULES
        start_blue = len(contours_red) + 1
        for i, cnt in enumerate(contours_blue, start=start_blue):
            area = cv2.contourArea(cnt)
            if area > min_area:
                subcolonias = procesar_subcolonias(img_original, cnt, ColonyType.BLUE, i)
                count_blue += max(1, subcolonias)
                total_colonias += max(1, subcolonias)

                (x, y), radius = cv2.minEnclosingCircle(cnt)
                colonies_data.append({
                    "id": i,
                    "type": ColonyType.BLUE,
                    "position": {"x": int(x), "y": int(y)},
                    "radius": int(radius),
                    "area": float(area),
                    "subcolonies": subcolonias
                })

        # Procesar colonias CAFÉ
        start_brown = len(contours_red) + len(contours_blue) + 1
        for i, cnt in enumerate(contours_brown, start=start_brown):
            area = cv2.contourArea(cnt)
            if area > min_area:
                subcolonias = procesar_subcolonias(img_original, cnt, ColonyType.BROWN, i)
                count_brown += max(1, subcolonias)
                total_colonias += max(1, subcolonias)

                (x, y), radius = cv2.minEnclosingCircle(cnt)
                colonies_data.append({
                    "id": i,
                    "type": ColonyType.BROWN,
                    "position": {"x": int(x), "y": int(y)},
                    "radius": int(radius),
                    "area": float(area),
                    "subcolonies": subcolonias
                })

        # Mostrar resumen en la imagen (igual que en tu código)
        resumen_texto = f"Total: {total_colonias} (R:{count_red} A:{count_blue} C:{count_brown})"
        cv2.putText(img_original, resumen_texto, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Crear resultado estructurado
        result = AnalysisResult(
            total_colonies=total_colonias,
            red_colonies=count_red,
            blue_colonies=count_blue,
            brown_colonies=count_brown,
            colonies=[ColonyData(**c) for c in colonies_data]
        )

        return img_original, result

    except Exception as e:
        raise ValueError(f"Error procesando imagen: {str(e)}")


@app.post("/analyze", response_model=AnalysisResult)
async def analyze_colonies(file: UploadFile = File(...)):
    """Endpoint para analizar imágenes JPG de colonias bacterianas"""
    try:
        # Verificar que sea JPG
        if not file.filename.lower().endswith(('.jpg', '.jpeg')):
            raise HTTPException(400, detail="Solo se aceptan imágenes en formato JPG")

        # Leer contenido
        contents = await file.read()

        processed_img, result = procesar_imagen_jpg(contents)

        output_filename = f"processed_{file.filename}"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        cv2.imwrite(output_path, processed_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        response = result.dict()
        response["processed_image_url"] = f"/results/{output_filename}"

        return JSONResponse(content=response)

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(400, detail=str(e))
    except Exception as e:
        raise HTTPException(500, detail=f"Error en el servidor: {str(e)}")


@app.get("/results/{filename}")
async def get_processed_image(filename: str):
    try:
        filename = "processed_" + filename + ".jpg"
        print(filename)
        file_path = os.path.join(OUTPUT_DIR, filename)
        if os.path.exists(file_path):
            return FileResponse(file_path, media_type="image/jpeg")
        raise HTTPException(404, detail="Imagen no encontrada")
    except Exception as e:
        raise HTTPException(500, detail=f"Error al recuperar imagen: {str(e)}")


@app.get("/")
async def health_check():
    return {"status": "active", "message": "API de análisis de colonias bacterianas (JPG)"}


@app.delete("/results/{filename}")
async def delete_processed_image(filename: str):
    """Eliminar una imagen procesada específica"""
    try:
        filename = "processed_" + filename + ".jpg"
        file_path = os.path.join(OUTPUT_DIR, filename)

        if not os.path.exists(file_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Imagen no encontrada"
            )

        os.remove(file_path)
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"message": f"Imagen {filename} eliminada correctamente"}
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al eliminar imagen: {str(e)}"
        )


@app.delete("/results/")
async def delete_all_processed_images():
    """Eliminar todas las imágenes procesadas"""
    try:
        deleted_files = []
        for filename in os.listdir(OUTPUT_DIR):
            file_path = os.path.join(OUTPUT_DIR, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    deleted_files.append(filename)
            except Exception as e:
                continue

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "message": "Eliminación completada",
                "deleted_files": deleted_files,
                "total_deleted": len(deleted_files)
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al eliminar imágenes: {str(e)}"
        )


@app.get("/results/")
async def list_processed_images():
    """Listar todas las imágenes procesadas disponibles"""
    try:
        files = []
        for filename in os.listdir(OUTPUT_DIR):
            file_path = os.path.join(OUTPUT_DIR, filename)
            if os.path.isfile(file_path):
                files.append({
                    "filename": filename,
                    "size": os.path.getsize(file_path),
                    "modified": os.path.getmtime(file_path)
                })

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "count": len(files),
                "files": sorted(files, key=lambda x: x["modified"], reverse=True)
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al listar imágenes: {str(e)}"
        )