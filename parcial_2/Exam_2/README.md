# Identificador de Objetos con OpenCV

| |  |  |
|----------|----------|----------|
| ![Resultado 1](results/Surf/Captura%20de%20pantalla%202024-10-23%20203947.png) | ![Resultado 2](results/Surf/Captura%20de%20pantalla%202024-10-23%20204205.png) | ![Resultado 3](results/Surf/Captura%20de%20pantalla%202024-10-23%20204336.png) |

|  |  |  |
|----------|----------|----------|
| ![Resultado 4](results/Surf/Captura%20de%20pantalla%202024-10-23%20204636.png) | ![Resultado 5](results/Surf/Captura%20de%20pantalla%202024-10-23%20205158.png) | ![Resultado 6](results/Surf/Captura%20de%20pantalla%202024-10-23%20205515.png) |


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

