# 🛣️ Detección de Huecos y Grietas en Pavimento

Este proyecto implementa un sistema de **Procesamiento Digital de Imágenes (PDI)** en **Python** que permite identificar **grietas e irregularidades en pavimento** a partir de videos.  
El objetivo es obtener un **reporte visual y numérico** del área y la longitud de las grietas detectadas a lo largo del video.

---

## 📌 Descripción General

El programa procesa cada cuadro del video, aplica filtros para reducir ruido, detecta bordes y encuentra contornos que correspondan a posibles grietas o huecos en el pavimento.  
Finalmente, genera un video con las zonas dañadas resaltadas y un archivo CSV con las métricas obtenidas.

---

## ⚙️ Requerimientos

Instalar las librerías necesarias:
```bash
pip install opencv-python numpy pandas matplotlib

```
---


## ▶️ Ejecución del Programa

1. Coloca tu video en la carpeta videos/ (por ejemplo circunvalar_udea.mp4).

2. Ejecuta el script principal:

```bash
python main.py
```

3. Se generarán los siguientes resultados:
   - `output/video_grietas.mp4`: video con las grietas resaltadas.
   - `output/reporte_grietas.csv`: archivo con los valores de área y longitud por frame.

Durante la ejecución también podrás visualizar en tiempo real el video con las detecciones.

---

## 🔍 Flujo de Trabajo del Algoritmo

El procesamiento sigue el siguiente flujo:

### 1. Lectura del video
Se carga el video frame por frame utilizando OpenCV (`cv2.VideoCapture`).

### 2. Preprocesamiento de imagen
- Conversión a escala de grises para reducir información innecesaria.
- Aplicación de un filtro Gaussiano para eliminar ruido de alta frecuencia.

### 3. Detección de bordes
Se utiliza el detector de **Canny** para resaltar los bordes potenciales que podrían corresponder a grietas o huecos.

### 4. Operaciones morfológicas
Se aplican operaciones de dilatación y erosión con un elemento estructurante rectangular para unir fragmentos discontinuos de las grietas.

### 5. Segmentación de grietas
- Se detectan contornos con `cv2.findContours`.
- Se filtran los contornos pequeños (ruido) y se calculan métricas geométricas:
  - **Área**: tamaño del daño detectado.
  - **Longitud**: perímetro del contorno.

### 6. Visualización y reporte
- Los contornos se dibujan sobre el frame original en color rojo.
- Se superponen los valores de área y longitud total sobre la imagen.
- Se almacenan los resultados en un DataFrame y luego en `reporte_grietas.csv`.
- Se genera un video con todos los frames procesados (`output/video_grietas.mp4`).

### 7. Análisis posterior (opcional)
- El archivo CSV permite realizar análisis estadísticos o graficar la evolución de las grietas por frame.
- Se puede calibrar el sistema para estimar medidas reales (en cm o m²) si se conoce la escala del video.

---

## 📊 Tecnologías Utilizadas

- **Python 3.x**
- **OpenCV** (procesamiento de imágenes y video)
- **NumPy** (operaciones matriciales)
- **Pandas** (generación del reporte CSV)

---

## 📁 Estructura del Proyecto
```
deteccion-de-huecos-en-pavimento/
│
├── videos/                 # Carpeta con el video original
│   └── circunvalar_udea.mp4
├── output/                 # Carpeta donde se guardan los resultados
│   ├── video_grietas.mp4
│   └── reporte_grietas.csv
├── main.py                 # Script principal con el algoritmo
├── requirements.txt        # Librerías necesarias
└── README.md               # Descripción del proyecto
```
## 👨‍💻 Autor
Proyecto desarrollado por Daniel Rúa
Como parte de las prácticas de Procesamiento Digital de Imágenes.
Universidad de Antioquia – 2025.