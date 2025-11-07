import cv2
import numpy as np
import pandas as pd
import os

def procesar_video(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    grietas_data = []
    frame_count = 0

    os.makedirs(output_dir, exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        edges = cv2.Canny(blur, 50, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        edges = cv2.dilate(edges, kernel, iterations=1)
        edges = cv2.erode(edges, kernel, iterations=1)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            length = cv2.arcLength(cnt, True)
            if area > 50:
                grietas_data.append({'frame': frame_count, 'area': area, 'length': length})
                cv2.drawContours(frame, [cnt], -1, (0,0,255), 2)

        # Guardar cada frame procesado en output
        cv2.imwrite(os.path.join(output_dir, f'frame_{frame_count:04d}.png'), frame)

    cap.release()
    df = pd.DataFrame(grietas_data)
    df.to_csv(os.path.join(output_dir, 'reporte_grietas.csv'), index=False)
    print("Proceso finalizado. Reporte generado en:", output_dir)

if __name__ == "__main__":
    procesar_video('videos/circunvalar_udea.mp4', 'output')
