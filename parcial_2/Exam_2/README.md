# Identificador de Objetos con OpenCV

<div style="display: flex; flex-wrap: wrap; justify-content: space-between;">
    <div style="width: 32%;"><img src="results/Surf/Captura de pantalla 2024-10-23 203947.png" alt="Resultado 1" style="width: 100%;"></div>
    <div style="width: 32%;"><img src="results/Surf/Captura de pantalla 2024-10-23 204205.png" alt="Resultado 2" style="width: 100%;"></div>
    <div style="width: 32%;"><img src="results/Surf/Captura de pantalla 2024-10-23 204336.png" alt="Resultado 3" style="width: 100%;"></div>
</div>

<div style="display: flex; flex-wrap: wrap; justify-content: space-between; margin-top: 20px;">
    <div style="width: 32%;"><img src="results/Surf/Captura de pantalla 2024-10-23 204636.png" alt="Resultado 4" style="width: 100%;"></div>
    <div style="width: 32%;"><img src="results/Surf/Captura de pantalla 2024-10-23 205158.png" alt="Resultado 5" style="width: 100%;"></div>
    <div style="width: 32%;"><img src="results/Surf/Captura de pantalla 2024-10-23 205515.png" alt="Resultado 6" style="width: 100%;"></div>
</div>

Resultados para SURF


## Descripción
Este proyecto implementa un identificador de objetos utilizando descriptores de imágenes en OpenCV. Se utilizan tres métodos principales de extracción de características:

- **BRISK (Binary Robust Invariant Scalable Keypoints)**: Rápido y eficiente para detectar puntos clave en imágenes.
- **SURF (Speeded Up Robust Features)**: Más robusto ante cambios de escala e iluminación.
- **HOG (Histogram of Oriented Gradients)**: Se enfoca en la estructura global de la imagen y se usa con un clasificador SVM.

El objetivo es identificar objetos en imágenes o en tiempo real mediante una cámara, comparando los descriptores con un conjunto de datos de referencia.

## Instalación
Para ejecutar este proyecto, es necesario instalar las siguientes dependencias:
```bash
pip install opencv-python numpy scikit-learn matplotlib
```

## Uso
Ejecutar el script con los siguientes parámetros:
```bash
python classifiers.py --i <ruta/dataset> --d <descriptor> --t 5 --r "(64,64)"
```
Donde:
- `--i`: Ruta a las imágenes de entrenamiento.
- `--d`: Descriptor a utilizar (`brisk`, `surf` o `hog`).
- `--t`: Umbral de coincidencias para la identificación.
- `--r`: Redimensionado opcional de las imágenes.

Para probar el reconocimiento en tiempo real con la cámara:
```bash
python classifiers.py --i <dataset> --d <hog> --t 5
```

## Resultados
Se evaluaron los tres descriptores con un conjunto de 7 objetos cotidianos. Los resultados obtenidos fueron:

| Descriptor | Precisión |
|------------|-----------|
| BRISK      | 16.14 matches promedio |
| SURF       | 54.28 matches promedio |
| HOG + SVM  | 92% precisión (pero con dificultad para algunas clases) |

SURF fue el descriptor más efectivo en general, seguido de BRISK y luego HOG, que requirió aumento de datos para mejorar la precisión.

## Conclusión
Este proyecto permitió comparar distintos métodos de identificación de imágenes con OpenCV. Mientras que SURF ofreció la mejor precisión, BRISK fue el más rápido. HOG con SVM mostró buenos resultados en clasificación, pero con limitaciones en la detección de objetos en video en tiempo real.

