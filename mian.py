import cv2
import numpy as np
from google.colab.patches import cv2_imshow # Import added here

def adjust_gamma(image, gamma=1.0):
    """
    Técnica no lineal para oscurecer medios tonos (grietas pálidas)
    sin perder la información general.
    gamma < 1.0 hace la imagen más oscura (resalta grietas claras).
    """
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def process_road_video_v3(video_path, output_path=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error al abrir video")
        return

    if output_path:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # --- AJUSTES DE ALTA SENSIBILIDAD ---
    # ROI: Corte del auto (20% abajo)
    CAR_MASK_PERCENT = 0.25

    # GAMMA: Valor clave para tus nuevas imágenes.
    # Un valor bajo (0.4 - 0.6) oscurece las grietas grises.
    GAMMA_VALUE = 0.5

    # BlackHat Kernel: Un poco más grande para asegurar que pille grietas largas
    BLACKHAT_KERNEL_SIZE = (20, 20)

    # Umbral: Lo bajamos aún más gracias a la limpieza previa
    THRESHOLD_VAL = 15

    # Filtros de forma
    MIN_AREA = 20         # Aceptamos fragmentos pequeños
    MAX_SOLIDITY = 0.6    # Las grietas son irregulares

    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    kernel_bh = cv2.getStructuringElement(cv2.MORPH_RECT, BLACKHAT_KERNEL_SIZE)
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    while True:
        ret, frame = cap.read()
        if not ret: break

        height, width = frame.shape[:2]

        # 1. Escala de grises
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # -------------------------------------------------------------
        # PASO CLAVE 1: FILTRO BILATERAL
        # -------------------------------------------------------------
        # Suaviza el "grano" del asfalto (reduce ruido) PERO mantiene
        # los bordes afilados de la grieta.
        # d=9 (diametro), sigmaColor=75, sigmaSpace=75
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)

        # -------------------------------------------------------------
        # PASO CLAVE 2: CORRECCIÓN GAMMA + CLAHE
        # -------------------------------------------------------------
        # Primero ecualizamos contraste local (CLAHE)
        enhanced = clahe.apply(filtered)
        # Luego aplicamos Gamma para hundir los grises en negro
        gamma_corrected = adjust_gamma(enhanced, gamma=GAMMA_VALUE)

        # -------------------------------------------------------------
        # PASO 3: Detección (BlackHat)
        # -------------------------------------------------------------
        blackhat = cv2.morphologyEx(gamma_corrected, cv2.MORPH_BLACKHAT, kernel_bh)

        # ROI y Pintura (Igual que antes)
        roi_mask = np.ones_like(gray, dtype=np.uint8) * 255
        cutoff_y = int(height * (1 - CAR_MASK_PERCENT))
        roi_mask[cutoff_y:, :] = 0

        # Detectar pintura sobre la imagen original (más brillante)
        _, paint_mask = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY)
        paint_mask = cv2.dilate(paint_mask, kernel_clean, iterations=4)

        # Aplicar máscaras
        blackhat[paint_mask > 0] = 0
        blackhat[roi_mask == 0] = 0

        # -------------------------------------------------------------
        # PASO 4: Binarización Adaptativa Híbrida
        # -------------------------------------------------------------
        # En lugar de un simple threshold, combinamos:
        # A. Lo que es oscurísimo (grieta profunda)
        _, t1 = cv2.threshold(blackhat, THRESHOLD_VAL, 255, cv2.THRESH_BINARY)

        # B. Lo que es medio oscuro (grieta fina) - Aumentamos sensibilidad
        # Normalizamos el blackhat para estirar el contraste al máximo 0-255
        normalized_bh = cv2.normalize(blackhat, None, 0, 255, cv2.NORM_MINMAX)
        _, t2 = cv2.threshold(normalized_bh, 35, 255, cv2.THRESH_BINARY)

        # Usamos la intersección lógica o unión ponderada?
        # Usaremos t2 (más sensible) pero la limpiaremos con operaciones morfológicas
        binary_cracks = t2

        # Limpieza de ruido "sal y pimienta" del asfalto
        binary_cracks = cv2.morphologyEx(binary_cracks, cv2.MORPH_OPEN, kernel_clean)

        # Detección de contornos
        cnts, _ = cv2.findContours(binary_cracks, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        vis_frame = frame.copy()
        cv2.line(vis_frame, (0, cutoff_y), (width, cutoff_y), (255, 0, 0), 2)

        detection_count = 0
        for c in cnts:
            area = cv2.contourArea(c)
            if area < MIN_AREA: continue

            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            if hull_area == 0: continue
            solidity = float(area) / hull_area

            # Filtro de relación de aspecto (Las grietas son alargadas)
            x,y,w,h = cv2.boundingRect(c)
            aspect_ratio = float(w)/h if h > 0 else 0

            # Una grieta no suele ser un cuadrado perfecto (aspect ratio ~ 1)
            # Aceptamos si es alargada O si tiene baja solidez
            if solidity < MAX_SOLIDITY or (aspect_ratio < 0.5 or aspect_ratio > 2.0):
                cv2.drawContours(vis_frame, [c], -1, (0, 0, 255), 2)
                detection_count += 1

        # VISUALIZACIÓN DIAGNÓSTICA
        # Verás: [Imagen Gamma (oscura) | Resultado Final]
        debug_view = cv2.cvtColor(gamma_corrected, cv2.COLOR_GRAY2BGR)
        stacked = np.hstack((debug_view, vis_frame))

        # Texto de estado
        cv2.putText(stacked, f"Grietas: {detection_count} | Gamma: {GAMMA_VALUE}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2_imshow(cv2.resize(stacked, (0,0), fx=0.5, fy=0.5)) # Changed cv2.imshow to cv2_imshow

        if output_path: out.write(vis_frame)
        if cv2.waitKey(1) == ord('q'): break

    cap.release()
    if output_path: out.release()
    cv2.destroyAllWindows()

process_road_video_v3('video_grietas.mp4')
print("Carga V3. Esta versión usa Filtro Bilateral y Gamma Correction para grietas tenues.")