import os
import argparse
import cv2
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import random

def augment_image(img):
    # Lista para almacenar las imágenes aumentadas
    augmented_images = []

    # 1. Rotación aleatoria
    angle = random.randint(-30, 30)  # Rotar entre -30 y 30 grados
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    rotated = cv2.warpAffine(img, M, (w, h))
    augmented_images.append(rotated)

    # 2. Cambio de brillo (solo aplica si la imagen tiene 3 canales de color)
    if len(img.shape) == 3 and img.shape[2] == 3:  # Verifica si la imagen es en color
        value = random.randint(-50, 50)  # Cambiar brillo entre -50 y 50 unidades
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.add(v, value)
        v = np.clip(v, 0, 255)
        hsv = cv2.merge((h, s, v))
        bright = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        augmented_images.append(bright)
    else:
        # Si es en escala de grises, aplica una modificación de brillo directamente
        value = random.randint(-50, 50)
        bright = cv2.add(img, value)
        bright = np.clip(bright, 0, 255).astype(np.uint8)
        augmented_images.append(bright)

    # 3. Inversión horizontal
    flipped = cv2.flip(img, 1)
    augmented_images.append(flipped)

    # 4. Zoom (recorte aleatorio)
    scale = random.uniform(0.8, 1.2)  # Escala de zoom entre 80% y 120%
    center_x, center_y = int(w / 2), int(h / 2)
    new_w, new_h = int(w * scale), int(h * scale)

    left = max(center_x - new_w // 2, 0)
    right = min(center_x + new_w // 2, w)
    top = max(center_y - new_h // 2, 0)
    bottom = min(center_y + new_h // 2, h)

    cropped = img[top:bottom, left:right]
    zoomed = cv2.resize(cropped, (w, h))
    augmented_images.append(zoomed)

    # 5. Añadir ruido gaussiano
    noise = np.random.normal(0, 25, img.shape).astype(np.uint8)
    noisy_img = cv2.add(img, noise)
    augmented_images.append(noisy_img)

    return augmented_images

def augment_dataset(images_by_class):
    augmented_dataset = {}
    for class_name, images in images_by_class.items():
        augmented_images = []
        for img in images:
            augmented_images.extend(augment_image(img))
        augmented_dataset[class_name] = images + augmented_images  # Agregar aumentaciones a originales
    return augmented_dataset

# Función para crear el descriptor HOG
def create_hog_descriptor(winSize=(64, 64), blockSize=(16, 16), blockStride=(8, 8),
                          cellSize=(8, 8), nbins=9):
    return cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)

# Función para cargar imágenes y calcular descriptores
def load_images(global_path, resize=None):
    class_names = os.listdir(global_path)
    data = {}
    for class_name in class_names:
        class_path = os.path.join(global_path, class_name)
        images = os.listdir(class_path)
        class_images = []  
        for image in images:
            image_path = os.path.join(class_path, image)
            img = cv2.imread(image_path, 0)
            if resize is not None:
                img = cv2.resize(img, resize)
            class_images.append(img)
        data[class_name] = class_images
    return data, class_names

# Función para cargar el descriptor
def load_descriptor(descriptor):
    if descriptor == "brisk":
        return cv2.BRISK_create()
    elif descriptor == "surf":
        return cv2.xfeatures2d.SURF_create()
    elif descriptor == "hog":
        return create_hog_descriptor()
    else:
        raise ValueError("Descriptor not found")

# Función para encontrar descriptores (se añadirá la lógica de SVM para HOG)
def find_descriptors(images_by_class, descriptor, descriptor_name):
    desList = []
    if descriptor_name == "hog":
        hog_descriptors = []
        labels = []
        class_index = 0
        for class_name, images in images_by_class.items():
            for img in images:
                des = descriptor.compute(img).flatten()
                hog_descriptors.append(des)
                labels.append(class_index)
            class_index += 1
        # Entrenar el clasificador SVM con HOG
        clf, scaler = train_svm(hog_descriptors, labels)
        return clf, scaler  # Devolver el clasificador SVM y el escalador
    else:
        for class_name, images in images_by_class.items():
            class_descriptors = []
            for img in images:
                kp, des = descriptor.detectAndCompute(img, None)
                if des is not None:
                    class_descriptors.append(des)
            desList.append(class_descriptors)
        return desList

# Función para entrenar SVM (específico para HOG)
def train_svm(hog_descriptors, labels):
    scaler = StandardScaler()
    hog_descriptors_scaled = scaler.fit_transform(hog_descriptors)
    X_train, X_test, y_train, y_test = train_test_split(hog_descriptors_scaled, labels, test_size=0.2, random_state=42)
    clf = svm.SVC(kernel='linear', C=1.0)
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    print(f"Precisión del modelo SVM: {accuracy:.2f}")
    return clf, scaler

# Función para identificar una imagen de entrada
def find_ID(img, desList, descriptor, class_names, threshold=5, descriptor_name="brisk", clf=None, scaler=None):
    if descriptor_name == "hog":
        # Para HOG, usar el clasificador SVM entrenado
        descriptor = descriptor.compute(img).flatten().reshape(1, -1)
        descriptor_scaled = scaler.transform(descriptor)
        label = clf.predict(descriptor_scaled)
        return class_names[label[0]], None
    else:
        # Mantener la lógica original para BRISK y SURF
        kp, des = descriptor.detectAndCompute(img, None)
        bf = cv2.BFMatcher()
        matches_list = []
        try:
            for class_descriptors in desList:
                class_match_couts = []
                for des2 in class_descriptors:
                    matches = bf.knnMatch(des, des2, k=2)
                    good = []
                    for m, n in matches:
                        if m.distance < 0.75 * n.distance:
                            good.append([m])
                    class_match_couts.append(len(good))
                matches_list.append(max(class_match_couts))
        except:
            pass
        if len(matches_list) != 0 and max(matches_list) > threshold:
            best_match_index = matches_list.index(max(matches_list))
            return class_names[best_match_index], matches_list[best_match_index]
        else:
            return (None, None)
        
def parse_tuple(arg):
    return tuple(map(int, arg.strip("()").split(',')))

# Función principal
if __name__ == "__main__":
    arg = argparse.ArgumentParser()
    arg.add_argument("--i", type=str, required=True, help="Path to the image")
    arg.add_argument("--d", type=str, required=True, help="Descriptor to use")
    arg.add_argument("--t", type=float, required=True, help="Threshold to use")
    arg.add_argument("--r", type=str, required=False, help="resize")
    args = vars(arg.parse_args())

    image_path = args["i"]
    descriptor_name_alg = args["d"]
    threshold = args["t"]
    aux_resize = parse_tuple(args["r"]) if args["r"] else None
    resize = aux_resize if aux_resize else None

    images_by_class, class_names = load_images(image_path, resize=resize)
    print(f'Total clases {len(images_by_class)}')

    images_by_class = augment_dataset(images_by_class)

    descriptor = load_descriptor(descriptor_name_alg)

    if descriptor_name_alg == "hog":
        clf, scaler = find_descriptors(images_by_class, descriptor, descriptor_name_alg)
    else:
        desList = find_descriptors(images_by_class, descriptor, descriptor_name_alg)

    print(f'Total descriptores: {sum(len(class_des) for class_des in desList)}' if descriptor_name_alg != "hog" else "SVM entrenado con HOG")

    cap = cv2.VideoCapture(0)
    while True:
        succes, img2 = cap.read()
        imgOriginal = img2.copy()
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        if resize is not None: 
            img2 = cv2.resize(img2, resize)

        if descriptor_name_alg == "hog":
            class_name, matches = find_ID(img2, None, descriptor, class_names, threshold=threshold, descriptor_name=descriptor_name_alg, clf=clf, scaler=scaler)
        else:
            class_name, matches = find_ID(img2, desList, descriptor, class_names, threshold=threshold)

        if class_name is not None:
            cv2.putText(imgOriginal, f'{class_name}, t:{threshold}, m:{matches}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Image", imgOriginal)
        if cv2.waitKey(1) == ord('q'):
            break
