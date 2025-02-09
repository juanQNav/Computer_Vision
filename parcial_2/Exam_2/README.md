# Object Identifier with OpenCV

| |  |  |
|----------|----------|----------|
| ![Result 1](results/Surf/Captura%20de%20pantalla%202024-10-23%20203947.png) | ![Result 2](results/Surf/Captura%20de%20pantalla%202024-10-23%20204205.png) | ![Result 3](results/Surf/Captura%20de%20pantalla%202024-10-23%20204336.png) |

|  |  |  |
|----------|----------|----------|
| ![Result 4](results/Surf/Captura%20de%20pantalla%202024-10-23%20204636.png) | ![Result 5](results/Surf/Captura%20de%20pantalla%202024-10-23%20205158.png) | ![Result 6](results/Surf/Captura%20de%20pantalla%202024-10-23%20205515.png) |

Results for SURF

## Description
This project implements an object identifier using image descriptors in OpenCV. Three main feature extraction methods are used:

- **BRISK (Binary Robust Invariant Scalable Keypoints)**: Fast and efficient for detecting key points in images.
- **SURF (Speeded Up Robust Features)**: More robust to scale and lighting changes.
- **HOG (Histogram of Oriented Gradients)**: Focuses on the global structure of the image and is used with an SVM classifier.

The goal is to identify objects in images or in real-time via a camera by comparing descriptors with a reference dataset.

## Installation
To run this project, install the following dependencies:
```bash
pip install opencv-python numpy scikit-learn matplotlib
```

## Usage
Run the script with the following parameters:
```bash
python classifiers.py --i <path/dataset> --d <descriptor> --t 5 --r "(64,64)"
```
Where:
- `--i`: Path to the training images.
- `--d`: Descriptor to use (`brisk`, `surf`, or `hog`).
- `--t`: Matching threshold for identification.
- `--r`: Optional image resizing.

To test real-time recognition with a camera:
```bash
python classifiers.py --i <dataset> --d <hog> --t 5
```

## Results
The three descriptors were evaluated with a set of 7 everyday objects. The results obtained were:

| Descriptor | Accuracy |
|------------|-----------|
| BRISK      | 16.14 average matches |
| SURF       | 54.28 average matches |
| HOG + SVM  | 92% accuracy (but struggled with some classes) |

SURF was the most effective descriptor overall, followed by BRISK, and then HOG, which required data augmentation to improve accuracy.

## Conclusion
This project allowed for a comparison of different image identification methods using OpenCV. While SURF provided the best accuracy, BRISK was the fastest. HOG with SVM showed good classification results but had limitations in real-time object detection in video.